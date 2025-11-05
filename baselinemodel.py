import re
import pandas as pd
from collections import Counter

# ======================================================
# 1. Keyword dictionaries for each supported language
# ======================================================
KEYWORDS = {
    "python": [
        "def", "import", "lambda", "self", "print", "in", "not", "and", "or",
        "elif", "except", "as", "from", "with", "yield", "async", "await"
    ],
    "java": [
        "public", "private", "protected", "class", "interface", "implements",
        "extends", "System.out", "println", "new", "import", "package",
        "static", "void", "int", "String", "boolean"
    ],
    "cpp": [
        "#include", "std::", "cout", "cin", "endl", "->", "::", "template",
        "typename", "using", "namespace", "int", "char", "bool", "void", "new"
    ],
    "javascript": [
        "function", "var", "let", "const", "=>", "console.log", "import", "export",
        "async", "await", "document", "window", "return", "class", "this"
    ],
    "mysql": [
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN",
        "CREATE", "TABLE", "DROP", "DATABASE", "VALUES", "INTO", "ALTER",
        "PRIMARY", "KEY", "FOREIGN"
    ],
    "verilog": [
        "module", "endmodule", "input", "output", "reg", "wire", "assign",
        "always", "posedge", "negedge", "begin", "end", "initial", "parameter"
    ]
}

# ======================================================
# 2. Token-based scoring function
# ======================================================
def score_language(code_snippet: str, lang: str) -> float:
    """Count keyword matches for a given language (case-insensitive)."""
    keywords = KEYWORDS[lang]
    score = 0
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        matches = len(re.findall(pattern, code_snippet, flags=re.IGNORECASE))
        score += matches
    return score

# ======================================================
# 3. Prediction function
# ======================================================
def predict_language(code_snippet: str) -> str:
    """Predict the most likely programming language for a given code snippet."""
    snippet_clean = code_snippet.strip()
    if not snippet_clean:
        return "unknown"

    scores = Counter()

    for lang in KEYWORDS:
        lang_score = score_language(snippet_clean, lang)
        # normalize by number of keywords to avoid bias for larger lists
        norm_score = lang_score / (len(KEYWORDS[lang]) or 1)
        scores[lang] = norm_score

    # select language with highest normalized score
    top_lang, top_score = scores.most_common(1)[0]
    if top_score == 0:
        return "unknown"
    return top_lang

# ======================================================
# 4. Baseline model evaluation
# ======================================================
VAL_PATH = "val_snippets.csv"
df = pd.read_csv(VAL_PATH)

total, correct = 0, 0
per_lang_counts = Counter()
per_lang_correct = Counter()

for _, row in df.iterrows():
    code = str(row["code"])
    true_lang = str(row["language"]).lower()
    pred_lang = predict_language(code)

    total += 1
    per_lang_counts[true_lang] += 1
    if pred_lang == true_lang:
        correct += 1
        per_lang_correct[true_lang] += 1

overall_acc = correct / total * 100

print(f"Baseline Keyword Model Accuracy: {overall_acc:.2f}%\n")

print("Per-language accuracy:")
for lang in sorted(per_lang_counts.keys()):
    acc = per_lang_correct[lang] / per_lang_counts[lang] * 100
    print(f"  {lang:<12s} {acc:5.2f}% ({per_lang_correct[lang]}/{per_lang_counts[lang]})")