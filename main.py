import torch
import sys
import os
import json
from config import DEVICE, WEIGHTS_FILE
from tokenizer import Tokenizer
from dataset import load_source_texts, build_dialogue
from model import ChatGPT
from train import train
from chat import chat

torch.manual_seed(42)

CHAT_ONLY  = "--chat" in sys.argv
VOCAB_FILE = "model/vocab.json"

os.makedirs("model", exist_ok=True)

if CHAT_ONLY:
    if not os.path.exists(WEIGHTS_FILE):
        print("No saved weights found! Run without --chat first to train.")
        exit()
    if not os.path.exists(VOCAB_FILE):
        print("No vocab found! Run without --chat first to train.")
        exit()

    with open(VOCAB_FILE, "r") as f:
        saved = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.stoi      = saved
    tokenizer.itos      = {int(i): w for w, i in saved.items()}
    tokenizer.vocab_size = len(saved)

    source_texts     = load_source_texts("src") if os.path.exists("src") else {}
    full_text        = " ".join(source_texts.values()) if source_texts else ""

    model = ChatGPT(vocab_size=tokenizer.vocab_size).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Vocab size : {tokenizer.vocab_size} tokens")
    print(f"Model      : {params:,} parameters (~{params*4/1e6:.1f} MB)\n")
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    print("Loaded saved weights — skipping training!\n")

else:
    source_texts      = load_source_texts("src")
    corpus, full_text = build_dialogue(source_texts)

    tokenizer = Tokenizer()
    tokenizer.build(corpus)

    print(f"Vocab size : {tokenizer.vocab_size} tokens")
    print(f"Corpus     : {len(tokenizer.encode(corpus)):,} tokens\n")

    with open(VOCAB_FILE, "w") as f:
        json.dump(tokenizer.stoi, f)
    print(f"Vocab saved → {VOCAB_FILE}")

    data  = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    model = ChatGPT(vocab_size=tokenizer.vocab_size).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters (~{params*4/1e6:.1f} MB)\n")

    model._train_data = data
    train(model)
    model.load_state_dict(torch.load(WEIGHTS_FILE))

chat(model, tokenizer, full_text, source_texts_dict=source_texts)