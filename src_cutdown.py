import os

SRC_DIR    = "src"
TARGET_MAX = 50000  #Change this to the maximum number of characters you want to allow for each file

for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(SRC_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) > TARGET_MAX:
        trimmed = text[:TARGET_MAX].rsplit(".", 1)[0] + "."
        with open(path, "w", encoding="utf-8") as f:
            f.write(trimmed)
        print(f"{fname}: {len(text):,} → {len(trimmed):,} chars (trimmed)")
    else:
        print(f"{fname}: {len(text):,} chars (ok)")

print(f"\nDone! All files have been cut down to {TARGET_MAX} characters or less.")