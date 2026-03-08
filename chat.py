import torch
import re
from config import BLOCK_SIZE, TEMPERATURE, TOP_K, MAX_NEW_TOKENS, DEVICE, SYSTEM_PROMPT
from torch.nn import functional as F


def get_relevant_chunk(query, source_texts_dict, chunk_size=80, overlap=20):
    import math
    query_words = set(re.findall(r"[\w']+", query.lower()))

    # build IDF — rarer words across all chunks score higher
    all_text = " ".join(source_texts_dict.values())
    all_words = re.findall(r"[\w']+", all_text.lower())
    total_chunks = max(1, len(all_words) // chunk_size)
    word_freq = {}
    for w in all_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    idf = {w: math.log(total_chunks / (1 + f)) for w, f in word_freq.items()}

    best, best_score = "", 0

    for fname, text in source_texts_dict.items():
        topic = re.sub(r'[\.txt_]', ' ', fname).lower()
        topic_words = set(re.findall(r"[\w']+", topic))
        topic_bonus = len(query_words & topic_words) * 3

        words, i = text.split(), 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            chunk_words = set(re.findall(r"[\w']+", chunk.lower()))
            matches = query_words & chunk_words
            s = sum(idf.get(w, 0) for w in matches) + topic_bonus
            if s > best_score:
                best_score, best = s, chunk
            i += chunk_size - overlap

    return best if best else list(source_texts_dict.values())[0][:chunk_size*6]


def generate(model, tokenizer, idx, temperature=TEMPERATURE, top_k=TOP_K, max_new_tokens=MAX_NEW_TOKENS):
    eos_id    = tokenizer.token_id("<eos>")
    eob_id    = tokenizer.token_id("</bot>")
    hum_id    = tokenizer.token_id("<human>")
    generated = []

    seen_tokens = {}
    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / temperature

        # repetition penalty
        for tok_id, count in seen_tokens.items():
            logits[0, tok_id] -= 0.5 * count

        v, _      = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, num_samples=1)
        tok_id    = next_tok.item()
        seen_tokens[tok_id] = seen_tokens.get(tok_id, 0) + 1

        if tok_id in (eos_id, eob_id, hum_id):
            break

        generated.append(tok_id)
        idx = torch.cat((idx, next_tok), dim=1)

    raw = tokenizer.decode(generated)
    for t in ["<s>", "</s>", "<human>", "</human>", "<bot>", "</bot>", "<eos>"]:
        raw = raw.replace(t, "").strip()
    return raw


def build_prompt(tokenizer, ctx_chunk, history, user_msg):
    sys_block = f"<s> {SYSTEM_PROMPT} </s>"
    ctx_block = f"<s> {ctx_chunk} </s>"
    prompt    = f"{sys_block} {ctx_block} {history} <human> {user_msg} </human> <bot>"

    while len(tokenizer.encode(prompt)) > BLOCK_SIZE - 10 and "<eos>" in history:
        cut     = history.index("<eos>") + len("<eos>")
        history = history[cut:].strip()
        prompt  = f"{sys_block} {ctx_block} {history} <human> {user_msg} </human> <bot>"

    return prompt, history


def chat(model, tokenizer, source_text, source_texts_dict=None):
    model.eval()
    print("\n" + "=" * 55)
    print("  NanoMind — Type your message and press Enter")
    print("  'quit' to exit  |  'reset' to clear history")
    print("=" * 55 + "\n")

    history = ""

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() == "quit":
            print("Goodbye!")
            break
        if user_msg.lower() == "reset":
            history = ""
            print("(conversation cleared)\n")
            continue

        src = source_texts_dict if source_texts_dict else {"data": source_text}
        ctx_chunk = get_relevant_chunk(user_msg, src)
        prompt, history = build_prompt(tokenizer, ctx_chunk, history, user_msg.lower())
        idx             = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        reply           = generate(model, tokenizer, idx)

        print(f"Bot: {reply}\n")

        history += f"<human> {user_msg} </human> <bot> {reply} </bot> <eos> "