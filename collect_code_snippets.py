from datasets import load_dataset
import pandas as pd
import re
import os

# ========== CONFIGURATION ==========
LANGUAGES = ["python", "cpp", "java", "mysql", "ruby", "go", "javascript", "verilog"]
SNIPPETS_PER_LANG = 500
MIN_LEN, MAX_LEN = 50, 1000
OUTPUT_CSV = "code_snippets.csv"
os.environ["HF_TOKEN"] = "insert_token" #took mine out
# ===================================

# Map user language names to the actual folder names used in The Stack
LANGUAGE_FOLDER_MAP = {
    "python": "python",
    "cpp": "c++",
    "java": "java",
    "mysql": "sql",
    "ruby": "ruby",
    "go": "go",
    "javascript": "javascript",
    "verilog": "veriloghdl"
}


def clean_code(code: str) -> str:
    """Remove comments and excessive whitespace."""
    if not code:
        return ""
    code = code.strip()
    code = re.sub(r"\n\s*\n+", "\n", code)
    code = re.sub(r"(?m)^\s*(#|//).*?$", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return code


def load_snippets_for_language(lang: str, n: int):
    """Load and clean code snippets for a given language."""
    folder = LANGUAGE_FOLDER_MAP.get(lang)
    if not folder:
        print(f"No folder mapping for {lang}, skipping.")
        return []

    print(f"Loading {n} snippets for: {lang} (folder: {folder})")
    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{folder}",
            split="train",
            streaming=True,
            token=os.environ["HF_TOKEN"]
        )
    except Exception as e:
        print(f"Skipping {lang}: cannot access folder '{folder}' - {e}")
        return []

    samples = []
    for i, sample in enumerate(ds):
        try:
            code = clean_code(sample.get("content", ""))
            if MIN_LEN < len(code) < MAX_LEN:
                samples.append({"code": code, "language": lang})
            if len(samples) >= n:
                break
        except Exception as e:
            print(f"Skipping sample {i}: {e}")
            continue

    print(f"Collected {len(samples)} valid snippets for {lang}")
    return samples


def main():
    all_data = []
    for lang in LANGUAGES:
        all_data.extend(load_snippets_for_language(lang, SNIPPETS_PER_LANG))

    print(f"Total snippets collected: {len(all_data)}")
    if not all_data:
        print("No snippets collected. Check token or language mapping.")
        return

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
