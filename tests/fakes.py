"""Offline test doubles: deterministic embeddings and a scripted LLM."""
import hashlib
import re

import numpy as np

from src.llm_service import LLMProvider


def fake_embed(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (out / norms).astype("float32")


class FakeLLM(LLMProvider):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, system="", json_mode=False, temperature=0.2):
        self.prompts.append(prompt)
        return self.replies.pop(0)
