#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import math
import glob
from typing import List, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

def parse_args():
    p = argparse.ArgumentParser(description="Generate static embeddings (Qwen3-0.6B Embedding) and text pairs.")
    p.add_argument("--data_dir", type=str, default="dataset/data/trival_QA",
                   help="Directory containing JSON files (TrivialQA-like).")
    p.add_argument("--output_dir", type=str, default="qwen3_embeddings",
                   help="Output directory for vectors.bin and metadata.txt")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                   help="Hugging Face model id for Qwen3 0.6B embedding")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device to run on")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding")
    p.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length (tokens)")
    p.add_argument("--chunk_size", type=int, default=800, help="Traditional chunk length (chars)")
    p.add_argument("--chunk_overlap", type=int, default=200, help="Traditional chunk overlap (chars)")
    p.add_argument("--target_dim", type=int, default=384,
                   help="Target dimension. If >0 and < model_dim, apply deterministic random projection")
    p.add_argument("--seed", type=int, default=42, help="Random seed for projection")
    return p.parse_args()


def load_json_texts(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []

    # 优先：TrivialQA 顶层 "docs"
    if isinstance(data, dict) and "docs" in data and isinstance(data["docs"], list):
        for d in data["docs"]:
            if isinstance(d, str) and d.strip():
                texts.append(d)
        return texts

    # 兼容：如果数据是 samples 且每个 sample 内自带 "documents"
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
    else:
        items = [data]

    for item in items:
        if isinstance(item, dict) and "documents" in item and isinstance(item["documents"], list):
            for d in item["documents"]:
                if isinstance(d, str) and d.strip():
                    texts.append(d)
    return texts

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    if chunk_overlap < 0:
        chunk_overlap = 0
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= n:
            break
        i += step
    return chunks


def collect_all_chunks(data_dir: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    all_chunks: List[str] = []
    for fp in files:
        texts = load_json_texts(fp)
        for t in texts:
            if not t or not isinstance(t, str):
                continue
            for ch in chunk_text(t, chunk_size, chunk_overlap):
                if ch.strip():
                    all_chunks.append(ch)
    return all_chunks


@torch.no_grad()
def embed_texts(texts: List[str],
                tokenizer: AutoTokenizer,
                model: AutoModel,
                device: torch.device,
                batch_size: int,
                max_length: int) -> np.ndarray:
    vecs: List[np.ndarray] = []
    model.eval()
    total_batches = math.ceil(len(texts) / max_length) if batch_size <= 0 else math.ceil(len(texts) / batch_size)
    rng = range(0, len(texts), batch_size)
    iterator = rng if _tqdm is None else _tqdm(rng, total=total_batches, desc="Embedding", unit="batch")
    for i in iterator:
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        # mean pooling with attention mask
        last_hidden = out.last_hidden_state  # [B, T, H]
        attn = enc.get("attention_mask", None)  # [B, T]
        if attn is None:
            # fallback to CLS (index 0)
            pooled = last_hidden[:, 0, :]
        else:
            mask = attn.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)        # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)        # [B, 1]
            pooled = summed / counts                        # [B, H]

        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs.append(pooled.cpu().numpy().astype(np.float32))
    if len(vecs) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(vecs)


def make_projection(model_dim: int, target_dim: int, seed: int) -> np.ndarray:
    if target_dim <= 0 or target_dim >= model_dim:
        return None
    rng = np.random.default_rng(seed)
    # 高斯随机投影并做列正交（QR）
    R = rng.standard_normal((model_dim, target_dim)).astype(np.float32)
    # QR 正交化，保证数值相对稳定
    Q, _ = np.linalg.qr(R)  # Q: (model_dim, model_dim)
    Q = Q[:, :target_dim].astype(np.float32)
    return Q


def sanitize_for_line(s: str) -> str:
    # 单行保存，避免换行破坏 metadata 对齐
    return " ".join(s.splitlines()).strip()


def save_vectors_bin(file_path: str, vectors: np.ndarray):
    # 写入格式：int32 count, int32 dim, 后接 row-major float32
    count, dim = vectors.shape
    with open(file_path, "wb") as f:
        f.write(np.asarray([count], dtype=np.int32).tobytes())
        f.write(np.asarray([dim], dtype=np.int32).tobytes())
        f.write(vectors.astype(np.float32).tobytes())


def main():
    args = parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Info] Collecting texts from: {args.data_dir}")
    chunks = collect_all_chunks(args.data_dir, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("[Error] No texts found. Please check data_dir.", file=sys.stderr)
        sys.exit(1)
    print(f"[Info] Total chunks: {len(chunks)} (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")

    print(f"[Info] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)

    print("[Info] Embedding...")
    embs = embed_texts(chunks, tokenizer, model, device, args.batch_size, args.max_length)  # [N, H]
    model_dim = embs.shape[1]
    print(f"[Info] Model embedding dim: {model_dim}")

    # 目标维度处理（若指定且小于模型维度，则做可复现随机投影）
    target_dim = args.target_dim if args.target_dim and args.target_dim > 0 else model_dim
    if target_dim < model_dim:
        print(f"[Info] Projecting to target_dim={target_dim} (seed={args.seed})")
        P = make_projection(model_dim, target_dim, args.seed)  # [H, target]
        embs = embs @ P
        # 再次归一化
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    elif target_dim > model_dim:
        print(f"[Warn] target_dim({target_dim}) > model_dim({model_dim}); using model_dim.", file=sys.stderr)
        target_dim = model_dim

    # 保存 vectors.bin & metadata.txt
    vectors_path = os.path.join(args.output_dir, "vectors.bin")
    metadata_path = os.path.join(args.output_dir, "metadata.txt")

    print(f"[Info] Saving vectors to: {vectors_path}")
    save_vectors_bin(vectors_path, embs.astype(np.float32))

    print(f"[Info] Saving metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for s in chunks:
            f.write(sanitize_for_line(s) + "\n")

    print(f"[Done] vectors: {embs.shape[0]} x {embs.shape[1]}")
    print(f"[Done] Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()