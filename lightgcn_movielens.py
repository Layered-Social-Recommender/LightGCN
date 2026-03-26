"""
LightGCN Recommendation System – MovieLens Small
================================================
Fixes vs starter code:
  1. Correct per-user negative sampling (only from unseen items)
  2. Proper 80/20 train/test split (leave-one-out per user)
  3. Mini-batch DataLoader for BPR training
  4. Top-K recommendations with movie title mapping

Dataset path (adjust if needed):
  d:/SEM6/LightGCN/ml-latest-small/ml-latest-small/
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import LightGCN

# ─────────────────────────── reproducibility ────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────── dataset paths ──────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "ml-latest-small", "ml-latest-small")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")

# ─────────────────────────── hyper-parameters ───────────────────────────────
EMBEDDING_DIM  = 64
NUM_LAYERS     = 3
LR             = 0.001
BATCH_SIZE     = 1024
NUM_EPOCHS     = 20
TOP_K          = 10
MIN_RATING     = 4.0          # implicit positive threshold
SAMPLE_USER_ID = 0            # internal idx for demo recommendations


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data …")
    ratings = pd.read_csv(RATINGS_PATH)
    movies  = pd.read_csv(MOVIES_PATH)

    # Filter to implicit positives
    pos = ratings[ratings["rating"] >= MIN_RATING].copy()

    # Re-index users and items to contiguous 0-based integers
    pos["user"], user_idx = pd.factorize(pos["userId"])
    pos["item"], item_idx = pd.factorize(pos["movieId"])

    n_users = pos["user"].nunique()
    n_items = pos["item"].nunique()

    # Movie ID → title lookup (using original movieId)
    movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
    # internal item index → original movieId
    idx_to_movie_id = dict(enumerate(item_idx))
    # internal item index → title
    item_to_title = {i: movie_id_to_title.get(idx_to_movie_id[i], "Unknown")
                     for i in range(n_items)}

    print(f"  Users: {n_users}  |  Items (>={MIN_RATING} stars): {n_items}  |  Interactions: {len(pos)}")
    return pos, n_users, n_items, item_to_title


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT  (leave-one-out per user)
# ══════════════════════════════════════════════════════════════════════════════

def split_data(pos, n_users, n_items):
    """
    For each user with ≥ 2 positives, hold out 1 random item as the test item.
    Returns:
        train_pairs   – list of (user, item) tuples used for training
        user_pos_set  – dict {user: set of ALL pos items} for negative sampling
        test_items    – dict {user: held-out test item index}
    """
    # Group items by user
    user_items = pos.groupby("user")["item"].apply(list).to_dict()

    train_pairs  = []
    test_items   = {}
    user_pos_set = {}          # ALL positives per user (train + test)

    for u in range(n_users):
        items = user_items.get(u, [])
        user_pos_set[u] = set(items)
        if len(items) < 2:
            # Can't split; use all for training, skip from evaluation
            train_pairs.extend((u, i) for i in items)
            continue
        test_item = random.choice(items)
        test_items[u] = test_item
        for i in items:
            if i != test_item:
                train_pairs.append((u, i))

    print(f"  Train pairs : {len(train_pairs)}")
    print(f"  Test users  : {len(test_items)}")
    return train_pairs, user_pos_set, test_items


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_edge_index(train_pairs, n_users):
    """
    Build undirected bipartite edge_index from training interactions only.
    Item node IDs are offset by n_users so users and items are in the same space.
    """
    users = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    items = torch.tensor([i + n_users for _, i in train_pairs], dtype=torch.long)

    # Undirected: both (u→i) and (i→u)
    edge_index = torch.stack([
        torch.cat([users, items]),
        torch.cat([items, users])
    ], dim=0)
    return edge_index


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET & DATALOADER FOR BPR TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class BPRDataset(Dataset):
    """
    Each item is a training (user, pos_item) pair.
    Negative is sampled at __getitem__ time from the user's unseen items.
    """
    def __init__(self, train_pairs, user_pos_set, n_items):
        self.pairs        = train_pairs
        self.user_pos_set = user_pos_set
        self.n_items      = n_items
        self.all_items    = list(range(n_items))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, pos_item = self.pairs[idx]
        # Sample a negative item the user has NOT interacted with
        while True:
            neg_item = random.randint(0, self.n_items - 1)
            if neg_item not in self.user_pos_set[user]:
                break
        return user, pos_item, neg_item


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(model, loader, optimizer, edge_index, n_users, device):
    model.train()
    total_loss = 0.0

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = (pos_items + n_users).to(device)   # offset to node space
        neg_items = (neg_items + n_users).to(device)

        optimizer.zero_grad()

        # Get all node embeddings after LightGCN propagation
        emb = model.get_embedding(edge_index.to(device))   # shape: [n_users+n_items, dim]

        u_emb   = emb[users]
        pos_emb = emb[pos_items]
        neg_emb = emb[neg_items]

        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)

        # BPR loss: maximise score gap between positive and negative
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION  –  Recall@K  and  NDCG@K
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, edge_index, test_items, user_pos_set, n_users, n_items,
             device, k=10):
    model.eval()
    emb = model.get_embedding(edge_index.to(device))

    u_emb = emb[:n_users]                             # [n_users, dim]
    i_emb = emb[n_users:]                             # [n_items, dim]

    # Score matrix: [n_users, n_items]
    scores = torch.mm(u_emb, i_emb.t())

    recalls, ndcgs = [], []

    for user, test_item in test_items.items():
        score_u = scores[user].cpu()

        # Mask out all training positives (don't recommend already-seen items)
        for seen in user_pos_set[user]:
            if seen != test_item:               # but keep the test item visible
                score_u[seen] = -1e9

        top_k = torch.topk(score_u, k).indices.tolist()

        # Recall@K
        hit = int(test_item in top_k)
        recalls.append(hit)

        # NDCG@K
        if hit:
            rank = top_k.index(test_item) + 1   # 1-indexed
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)

    recall = float(np.mean(recalls))
    ndcg   = float(np.mean(ndcgs))
    return recall, ndcg


# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP-K RECOMMENDATIONS FOR A SAMPLE USER
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def recommend(model, edge_index, user_id, user_pos_set, n_users, n_items,
              item_to_title, device, k=10):
    model.eval()
    emb   = model.get_embedding(edge_index.to(device))
    u_emb = emb[user_id]                              # [dim]
    i_emb = emb[n_users:]                             # [n_items, dim]

    scores = torch.mv(i_emb, u_emb).cpu()

    # Mask items the user has already seen
    for seen in user_pos_set[user_id]:
        scores[seen] = -1e9

    top_k = torch.topk(scores, k).indices.tolist()

    print(f"\n{'─'*55}")
    print(f"  Top-{k} Recommendations for internal User ID {user_id}")
    print(f"{'─'*55}")
    for rank, item_idx in enumerate(top_k, 1):
        title = item_to_title.get(item_idx, "Unknown")
        print(f"  {rank:2d}. {title}")
    print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Load data
    pos, n_users, n_items, item_to_title = load_data()

    # 2. Split
    print("\nSplitting data …")
    train_pairs, user_pos_set, test_items = split_data(pos, n_users, n_items)

    # 3. Graph
    edge_index = build_edge_index(train_pairs, n_users)
    print(f"  Edge index shape: {edge_index.shape}")

    # 4. DataLoader
    dataset = BPRDataset(train_pairs, user_pos_set, n_items)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, drop_last=False)

    # 5. Model
    model = LightGCN(
        num_nodes    = n_users + n_items,
        embedding_dim= EMBEDDING_DIM,
        num_layers   = NUM_LAYERS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6. Training loop
    print(f"\nTraining LightGCN for {NUM_EPOCHS} epochs …")
    print(f"{'Epoch':>6}  {'BPR Loss':>10}")
    print("─" * 22)

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(model, loader, optimizer, edge_index, n_users, device)
        print(f"{epoch:>6}  {loss:>10.4f}")

    # 7. Evaluation
    print("\nEvaluating …")
    recall, ndcg = evaluate(model, edge_index, test_items, user_pos_set,
                            n_users, n_items, device, k=TOP_K)
    print(f"\n{'─'*35}")
    print(f"  Recall@{TOP_K}  : {recall:.4f}")
    print(f"  NDCG@{TOP_K}    : {ndcg:.4f}")
    print(f"{'─'*35}")

    # 8. Sample recommendations
    recommend(model, edge_index, SAMPLE_USER_ID, user_pos_set,
              n_users, n_items, item_to_title, device, k=TOP_K)

    print("Done! LightGCN training and evaluation complete.")


if __name__ == "__main__":
    main()
