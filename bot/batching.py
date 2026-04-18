# -*- coding: utf-8 -*-
"""
Batching and embeddings module for organizing data processing.
"""

import math
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


def _make_query_for_embed(row) -> str:
    opts = row.get("options", None)
    opts_txt = "" if (opts is None or len(opts)==0) else " " + " ".join(map(str, opts))
    return f"{row.get('context','')} {row.get('question','')}{opts_txt}"


def _compute_embeddings_e5_mistral(texts, cache_path=None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape[0] == len(texts):
                print(f"[embeddings] loaded cached embeddings from {cache_path}")
                return arr
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[embeddings] using intfloat/e5-mistral-7b-instruct on {device} dtype={torch_dtype}")
    tok = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct", use_fast=True)
    model_e = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct", torch_dtype=torch_dtype, device_map="auto" if device=="cuda" else None)
    model_e.eval()
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    texts_prefixed = [f"query: {t}" for t in texts]
    embs = []; bs = 32
    with torch.no_grad():
        for i in range(0, len(texts_prefixed), bs):
            chunk = texts_prefixed[i:i+bs]
            enc = tok(chunk, padding=True, truncation=True, max_length=4096, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model_e(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().to("cpu").float())
    emb = np.asarray(torch.cat(embs, dim=0).numpy())
    if cache_path is not None:
        np.save(cache_path, emb); print(f"[embeddings] saved to cache {cache_path}")
    return emb


def _build_batches_kmeans_balanced(embeddings: np.ndarray, batch_size: int, random_state: int = 0):
    N = embeddings.shape[0]; K = int(math.ceil(N / batch_size))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = embeddings / np.clip(norms, 1e-12, None)
    try:
        from cuml.cluster import KMeans as cuKMeans
        km = cuKMeans(n_clusters=K, init="k-means++", max_iter=100, random_state=random_state, verbose=0)
        labels = km.fit_predict(X).get()
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K, init="k-means++", random_state=random_state, batch_size=8192, n_init="auto")
        labels = km.fit_predict(X)
    idx_by_cluster = [np.where(labels == c)[0].tolist() for c in range(K)]
    centroids = np.vstack([X[idx].mean(axis=0) if len(idx) else np.zeros(X.shape[1]) for idx in idx_by_cluster])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    for c, idxs in enumerate(idx_by_cluster):
        if not idxs: continue
        scores = X[idxs] @ centroids[c]
        idx_by_cluster[c] = [i for _, i in sorted(zip(scores, idxs))]
    batches, carry = [], []
    for idxs in idx_by_cluster:
        for i in range(0, len(idxs), batch_size):
            chunk = idxs[i:i+batch_size]
            if len(chunk) == batch_size: batches.append(chunk)
            else: carry.extend(chunk)
    for i in range(0, len(carry), batch_size):
        batches.append(carry[i:i+batch_size])
    return batches


def build_batches(df_in: pd.DataFrame, batch_size: int, embed_cache: Optional[str] = None, strategy: str = "kmeans"):
    N = len(df_in); indices = list(range(N))
    if strategy == "sequential":
        return [indices[i:i+batch_size] for i in range(0, N, batch_size)]
    if strategy == "random":
        rng = np.random.default_rng(0); rng.shuffle(indices)
        return [indices[i:i+batch_size] for i in range(0, N, batch_size)]
    texts = [_make_query_for_embed(row) for _, row in df_in.iterrows()]
    emb = _compute_embeddings_e5_mistral(texts, cache_path=embed_cache)
    return _build_batches_kmeans_balanced(emb, batch_size)