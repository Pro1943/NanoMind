# NanoMind 🧠

> "Tiny but thinks."

A Small Language Model (SLM) built from scratch in pure Python and PyTorch. Train it on any text, chat with it in your terminal. No GPUs. No cloud. No HuggingFace. Just Python.

Built by [@pro1943](https://github.com/pro1943) on an Intel Pentium 5405U with 4GB RAM.

---

## Features

- Transformer architecture built from scratch
- Word-level tokenizer with hardcoded special tokens
- RAG (Retrieval Augmented Generation) — finds the most relevant chunk from your source data at chat time
- Multi-file training — drop any `.txt` files into `src/` and it trains on all of them automatically
- Auto-saves best checkpoint during training
- `--chat` flag to skip training and jump straight to conversation
- Runs entirely on CPU

---

## Project Structure

```
NanoMind/
├── main.py           — entry point
├── config.py         — all settings, edit this to tune the model
├── tokenizer.py      — word-level tokenizer with special tokens
├── dataset.py        — loads all .txt files from src/, builds training corpus
├── model.py          — transformer architecture
├── train.py          — training loop with cosine LR scheduler
├── chat.py           — chat loop with RAG
├── src_cutdown.py    — trims src/ files to equal size before training
├── clean_wiki.py     — strips Wikipedia citation noise from txt files
├── model/
│   ├── nanoMind_best.pt  — saved weights (auto-generated)
│   └── vocab.json        — tokenizer vocab (auto-generated)
└── src/
    └── *.txt         — your training data goes here

```

---

## Quickstart

**1. Install dependencies**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

```

**2. Add training data**

Drop any `.txt` files into `src/`. The more focused and clean the text, the better.

Recommended: use the companion tool **WikiScrape** ([wiki-scrape.vercel.app](https://wiki-scrape.vercel.app/)) to download clean Wikipedia articles in one click.

**3. Train**

```bash
python main.py

```

**4. Chat**

```bash
python main.py --chat

```

---

## Chat Commands


| Command | Action                     |
| ------- | -------------------------- |
| `quit`  | Exit                       |
| `reset` | Clear conversation history |


---

## Configuration

All settings live in `config.py`:

```python
BATCH_SIZE     = 24      # training batch size
BLOCK_SIZE     = 64      # context window — higher = better memory, slower
MAX_ITERS      = 5000    # training steps
EVAL_EVERY     = 1000    # evaluate the model every 'n'th itteration
LEARN_RATE     = 8e-4    # learning rate
N_EMBED        = 128     # embedding dimension
N_HEAD         = 4       # attention heads
N_LAYER        = 4       # transformer blocks
DROPOUT        = 0.2     # regularization
TEMPERATURE    = 0.8     # generation randomness
DEVICE         = "cpu"   # the training device [chnage to "cuda" if you have a GPU]
TEMPERATURE = 0.8        # controls randomness: higher (e.g., 1.0) is creative/diverse, lower (e.g., 0.2) is focused/deterministic
TOP_K = 30               # limits next-token choices to the top 'k' most likely options to prevent nonsensical "long-tail" outputs
MAX_NEW_TOKENS = 80      # the maximum number of tokens (words/characters) the model will generate in a single response
```

---

## How It Works

```
Your .txt files in src/
        ↓
Word-level tokenizer builds vocab
        ↓
Transformer trains on dialogue-formatted corpus
        ↓
At chat time: RAG retrieves most relevant chunk
        ↓
Model generates response from that context

```

---

## Ideal Dataset


| Property  | Recommendation                |
| --------- | ----------------------------- |
| File size | 15k – 50k chars per file      |
| Format    | Clean plain text, no markdown |
| Topics    | Focused — one topic per file  |
| Sources   | Wikipedia articles work great |


Use `src_cutdown.py` to trim files to equal size before training.  
Use `clean_wiki.py` to strip citation noise from Wikipedia text.

---

## Hardware

NanoMind is designed for minimal hardware:


| Spec    | Minimum           |
| ------- | ----------------- |
| RAM     | 2 GB free         |
| CPU     | Any x64 processor |
| GPU     | Not required      |
| Storage | ~50 MB            |


Tested on: Intel Pentium 5405U, 4GB DDR4, no GPU, Windows 11.

---

## Companion Tool

**WikiScrape** — scrape and clean Wikipedia articles for training data in one click.  
[wiki-scrape.vercel.app](https://wiki-scrape.vercel.app/) · [@pro1943](https://github.com/pro1943)

---

## License

This project is licensed under the **NanoMind Open Use License**.  
Free to use, modify, and build on — just give credit.  
See [LICENSE]for full terms.

---

*Architecture inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).*