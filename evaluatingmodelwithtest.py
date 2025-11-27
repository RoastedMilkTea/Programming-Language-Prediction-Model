import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==========================
# Load Unseen Test Dataset
# ==========================
test_dataset = CodeDataset(
    "test_snippets.csv",
    vocab=train_dataset.vocab,   
    label_encoder=train_dataset.le 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)

# ==========================
# Evaluate Model
# ==========================
model.eval()
correct = 0
total = 0
per_language_correct = {}
per_language_total = {}

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        outputs = model(x_batch)
        preds = outputs.argmax(dim=1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        # Track per-language accuracy
        for label, pred in zip(y_batch, preds):
            lang = test_dataset.le.inverse_transform([label.item()])[0]
            if lang not in per_language_correct:
                per_language_correct[lang] = 0
                per_language_total[lang] = 0

            per_language_total[lang] += 1
            if label == pred:
                per_language_correct[lang] += 1

overall_acc = correct / total
print(f"Accuracy on Unseen Test Snippets: {overall_acc:.4f}")

# ==========================
# Plot Per-Language Accuracy
# ==========================
languages = list(per_language_correct.keys())
accs = [
    per_language_correct[lang] / per_language_total[lang]
    for lang in languages
]

plt.figure(figsize=(10,5))
plt.bar(languages, accs, color='skyblue')
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.xlabel("Language")
plt.title("Model Accuracy on Completely Unseen Code Snippets")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("unseen_results.png", dpi=300)
plt.show()