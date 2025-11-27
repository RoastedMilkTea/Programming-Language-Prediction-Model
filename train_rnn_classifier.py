import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# ==========================
# CONFIGURATION
# ==========================
TRAIN_PATH = "train_snippets.csv"
VAL_PATH   = "val_snippets.csv"

EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LR = 5e-4
EPOCHS = 15
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# DATA PREPARATION
# ==========================
def simple_tokenizer(code):
    """Basic tokenizer for code: split by non-alphanumeric characters."""
    tokens = re.findall(r"[A-Za-z_]+|\d+|[(){};.,=+\-*/<>]", code)
    return tokens

def code_tokenizer(code):
    tokens = re.findall(
        r'[A-Za-z_][A-Za-z_0-9]*|'           # identifiers / keywords
        r'\".*?\"|\'.*?\'|'                  # string / char literals
        r'==|!=|<=|>=|===|!==|'              # comparison
        r'->|=>|::|<<|>>|'                   # language-specific ops
        r'\+\+|--|\+=|-=|\*=|/=|%=|'         # assignment ops
        r'//.*?$|/\*.*?\*/|'                 # comments
        r'[(){}\[\];.,=+\-*/<>]|',           # punctuation
        code,
        flags=re.MULTILINE | re.DOTALL
    )
    return [t for t in tokens if t.strip() != '']



class CodeDataset(Dataset):
    def __init__(self, csv_path, vocab=None, label_encoder=None, build_vocab=False):
        df = pd.read_csv(csv_path)
        self.codes = df["code"].astype(str).tolist()
        self.labels = df["language"].tolist()

        # Encode labels
        if label_encoder is None:
            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(self.labels)
        else:
            self.le = label_encoder
            self.labels = self.le.transform(self.labels)

        # Build or reuse vocab
        if build_vocab:
            tokens = []
            for c in self.codes:
                tokens.extend(code_tokenizer(c))
            vocab = {"<PAD>": 0, "<UNK>": 1}
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        # Convert code snippets to index tensors
        self.data = []
        MAX_LEN = 200  # truncate long code samples to avoid exploding sequence length
        for code in self.codes:
            tokens = code_tokenizer(code)
            token_ids = [self.vocab.get(t, 1) for t in tokens][:MAX_LEN]  # trim sequence
            self.data.append(torch.tensor(token_ids, dtype=torch.long))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    codes, labels = zip(*batch)
    padded = pad_sequence(codes, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)


# ==========================
# MODEL DEFINITION
# ==========================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, outputs): 
        # outputs: (B, T, H)
        weights = torch.softmax(self.att(outputs).squeeze(-1), dim=1)  # (B, T)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)   # (B, H)
        return context

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                    bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.attention = Attention(hidden_dim*2)

    def forward(self, x):
        x = self.embedding(x)                    
        outputs, (h, c) = self.lstm(x)# h shape: (num_layers*2, B, H)

        context = self.attention(outputs)
        context = self.dropout(context)      
        # final = torch.cat((last_fw, last_bw), dim=1)  

        # final = self.dropout(final)

        out = self.fc(context)                  
        return out



# ==========================
# TRAINING LOOP
# ==========================
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_correct, total_samples, total_loss = 0, 0, 0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / len(train_loader)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_correct, val_samples, val_loss = 0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_samples += y_batch.size(0)

        val_acc = val_correct / val_samples
        val_loss = val_loss / len(val_loader)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, "
              f"Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}")

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history


# ==========================
# MAIN SCRIPT
# ==========================
if __name__ == "__main__":
    print("Loading datasets...")
    train_dataset = CodeDataset(TRAIN_PATH, build_vocab=True)
    val_dataset = CodeDataset(VAL_PATH, vocab=train_dataset.vocab,
                              label_encoder=train_dataset.le)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    vocab_size = len(train_dataset.vocab)
    num_classes = len(train_dataset.le.classes_)
    print(f"Vocab size: {vocab_size}, Classes: {num_classes}")

    model = RNNClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes,
                          num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # Plot learning curves
    plt.figure(figsize=(7,4))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('RNN Training vs Validation Accuracy')
    plt.tight_layout()
    plt.savefig('rnn_learning_curve.png', dpi=300)
    plt.show()
