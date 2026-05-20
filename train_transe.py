import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dim",      type=int,   default=64)
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--neg_ratio",    type=int,   default=1)
    p.add_argument("--gamma",        type=float, default=1.0)
    p.add_argument("--triples_path", type=str,   default="data/kg_triples.tsv")
    p.add_argument("--out_emb",      type=str,   default="pretrained/transe_entity_emb.npy")
    p.add_argument("--out_e2id",     type=str,   default="pretrained/transe_entity2id.json")
    return p.parse_args()


def load_triples(path: str) -> tuple:
    df = pd.read_csv(path, sep="\t")
    entities = sorted(set(df["head"]) | set(df["tail"]))
    relations = sorted(set(df["relation"]))
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}
    triples = [
        (e2id[h], r2id[r], e2id[t])
        for h, r, t in zip(df["head"], df["relation"], df["tail"])
    ]
    return triples, e2id, r2id


def main():
    args = parse_args()
    triples, e2id, r2id = load_triples(args.triples_path)
    n_e, n_r, d = len(e2id), len(r2id), args.emb_dim

    print(f"[TransE] entities: {n_e}, relations: {n_r}, emb_dim: {d}")

    # Uniform init in [-6/sqrt(d), 6/sqrt(d)] (Bordes et al. 2013)
    bound = 6.0 / (d ** 0.5)
    ent_emb = nn.Embedding(n_e, d)
    rel_emb = nn.Embedding(n_r, d)
    nn.init.uniform_(ent_emb.weight, -bound, bound)
    nn.init.uniform_(rel_emb.weight, -bound, bound)
    with torch.no_grad():
        rel_emb.weight.data = F.normalize(rel_emb.weight.data, p=2, dim=1)

    optimizer = torch.optim.SGD(
        list(ent_emb.parameters()) + list(rel_emb.parameters()),
        lr=args.lr,
    )

    triples_t = torch.tensor(triples, dtype=torch.long)  # [N, 3]
    n_triples = len(triples)

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_triples)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_triples, args.batch_size):
            batch = triples_t[perm[start : start + args.batch_size]]  # [B, 3]
            B = batch.shape[0]
            h_idx, r_idx, t_idx = batch[:, 0], batch[:, 1], batch[:, 2]

            # Build neg_ratio negatives per positive (50% head / 50% tail corruption)
            neg_h_list, neg_t_list = [], []
            for _ in range(args.neg_ratio):
                flip   = torch.rand(B) < 0.5
                rand_e = torch.randint(0, n_e, (B,))
                neg_h_list.append(torch.where(flip,  rand_e, h_idx))
                neg_t_list.append(torch.where(~flip, rand_e, t_idx))

            h_pos = h_idx.repeat(args.neg_ratio)
            r_pos = r_idx.repeat(args.neg_ratio)
            t_pos = t_idx.repeat(args.neg_ratio)
            neg_h = torch.cat(neg_h_list)
            neg_t = torch.cat(neg_t_list)

            # L1 margin ranking loss: max(0, gamma + d_pos - d_neg)
            pos_score = (ent_emb(h_pos) + rel_emb(r_pos) - ent_emb(t_pos)).abs().sum(1)
            neg_score = (ent_emb(neg_h) + rel_emb(r_pos) - ent_emb(neg_t)).abs().sum(1)
            loss = F.relu(args.gamma + pos_score - neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Constrain entity embeddings to unit sphere after each update
            with torch.no_grad():
                ent_emb.weight.data = F.normalize(ent_emb.weight.data, p=2, dim=1)

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 100 == 0:
            print(f"[TransE] epoch {epoch}/{args.epochs}, loss: {epoch_loss / n_batches:.4f}")

    # Final L2-normalize and save
    final_emb = F.normalize(ent_emb.weight.data, p=2, dim=1).detach().numpy()
    np.save(args.out_emb, final_emb)
    with open(args.out_e2id, "w") as f:
        json.dump(e2id, f)
    print(f"[TransE] saved: {args.out_emb}, {args.out_e2id}")


if __name__ == "__main__":
    main()
