import os

BATCH_SIZE     = 24
BLOCK_SIZE     = 64
MAX_ITERS      = 5000
EVAL_EVERY     = 1000
LEARN_RATE     = 8e-4
N_EMBED        = 128
N_HEAD         = 4
N_LAYER        = 4
DROPOUT        = 0.2
DEVICE         = "cpu"

TEMPERATURE    = 0.8
TOP_K          = 30
MAX_NEW_TOKENS = 80

SRC_DIR        = "src"
WEIGHTS_FILE   = "model/nanoMind_best.pt"

SYSTEM_PROMPT  = "You are a helpful and safe AI assistant . Answer clearly and stay on topic . If you cannot help , say : I cannot help with that ."
