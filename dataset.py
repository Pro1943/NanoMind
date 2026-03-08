import os
from config import SRC_DIR, SYSTEM_PROMPT

def load_source_texts(src_dir):
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
        sample_path = os.path.join(src_dir, "data.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Replace this with your own text.")
        print(f"Created {sample_path} — add your content and re-run!")
        exit()

    texts = {}
    for fname in os.listdir(src_dir):
        if fname.endswith(".txt"):
            path = os.path.join(src_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts[fname] = f.read().strip()

    if not texts:
        print(f"No .txt files found in {src_dir}!")
        exit()

    print(f"Loaded {len(texts)} file(s) from {src_dir}/:")
    for fname, text in texts.items():
        print(f"  {fname} — {len(text):,} chars")

    return texts

def chunk_text(text, chunk_size=80, overlap=20):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def wrap(human_text, bot_text, source_chunk=None):
    sys_block  = f"<s> {SYSTEM_PROMPT} </s>"
    ctx_block  = f"<s> {source_chunk} </s>" if source_chunk else ""
    human_block = f"<human> {human_text} </human>"
    bot_block   = f"<bot> {bot_text} </bot>"
    return f"{sys_block} {ctx_block} {human_block} {bot_block} <eos>\n"

def build_dialogue(source_texts):
    all_chunks = []
    full_text  = ""

    for text in source_texts.values():
        full_text += " " + text
        all_chunks.extend(chunk_text(text))

    full_text = full_text.strip()
    print(f"Total chunks (no limit): {len(all_chunks)}")

    CONVOS_STATIC = [
        ("hi",            "Hello ! How can I help you ?"),
        ("hello",         "Hello ! What would you like to know ?"),
        ("hey",           "Hello ! Ask me anything ."),
        ("how are you",   "Ready to help ! What do you want to ask ?"),
        ("what do you know", "I have studied the provided material . Ask me anything !"),
        ("help",          "You can ask me questions or request a summary about the material ."),
        ("thanks",        "You are welcome !"),
        ("thank you",     "Happy to help !"),
        ("bye",           "Goodbye !"),
        ("ok",            "What else would you like to know ?"),
        ("tell me more",  "What aspect would you like me to elaborate on ?"),
        ("interesting",   "There is more to explore . What else would you like to know ?"),
        ("wow",           "Pretty interesting ! What else would you like to know ?"),
        ("i don't know",  "That is okay ! Feel free to ask me anything about the material ."),
        ("what else",     "You can ask me to summarize , explain , or answer questions about the text ."),
        ("can you explain",         "What specifically would you like explained ?"),
        ("explain",                   "Based on the material :"),
        ("what is",                   "Based on the material :"),
        ("what are",                  "Based on the material :"),
        ("how does",                  "Based on the material :"),
        ("how do",                    "Based on the material :"),
        ("tell me about",             "Based on the material :"),
        ("describe",                  "Based on the material :"),
        ("define",                    "Based on the material :"),
        ("give me info",              "Based on the material :"),
        ("what do you know about",    "Based on the material :"),
    ]

    corpus = ""

    for chunk in all_chunks:
        chunk_words = " ".join(chunk.split()[:40])
        CONVOS_DYNAMIC = [
            ("summarize",              "Here is a summary : " + chunk_words),
            ("give me a summary",      "Here is a summary : " + chunk_words),
            ("summary please",         "Here is a summary : " + chunk_words),
            ("summary of the content", "Here is a summary : " + chunk_words),
            ("what is this about",     "This is about : " + chunk_words),
            ("what is the main topic", "The main topic is : " + chunk_words),
            ("tell me a fact",         "Here is a fact : " + chunk_words),
        ]
        for human, bot in CONVOS_STATIC + CONVOS_DYNAMIC:
            corpus += wrap(human, bot, source_chunk=chunk)

    for human, bot in CONVOS_STATIC:
        corpus += wrap(human, bot)

    corpus *= 2
    return corpus, full_text