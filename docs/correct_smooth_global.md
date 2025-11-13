# Global Item-User Embeddings with Correct-and-Smooth: Methodology and Rationale

This md dissects the approach in `correct_smooth_global.ipynb`: training a global item-user embedding model on sparse movie features, then refining per-user predictions via a Correct-and-Smooth (C+S) procedure over a similarity graph. The focus is on *why* each design choice was made and how it impacts performance and scalability.

## High-Level Pipeline

1. Load processed movie and rating data; map external IDs to contiguous indices.
2. Engineer sparse content features (overview TF-IDF, genres, keywords, adult flag, numeric signals).
3. Load a precomputed movie similarity graph and optionally symmetrize + normalize it.
4. Train a shared item encoder + user embeddings using binary labels (like vs. not like per user).
5. Precompute item embeddings.
6. For each user, produce base scores, apply correction on labeled items, then iterative smoothing over the graph.
7. Collect and save per-user metrics for baseline vs. C+S.

## Per-User Binary Labels

```python
ratings = ratings.groupby('userId', group_keys=False).apply(
    lambda df: df.assign(label=(df['rating'] >= df['rating'].mean()).astype(int))
)
```

Rationale:

- Users exhibit different scoring scales; personal mean centers the decision boundary.
- Avoids bias toward users who rate high or low globally.
- Dynamic thresholding adapts automatically to long-tailed rating distributions.

## Sparse Feature Engineering

Components:

- TF-IDF overview (semantic content).
- Genres & keywords via multi-label binarization (taxonomy signals).
- Adult flag (binary content filter).
- Numeric features (popularity, runtime, vote stats; `log1p` for counts to reduce skew).

Why sparse matrix:

- Memory efficiency for high-dimensional text features.
- Interoperable with scikit-learn transformers.
- Enables lazy row extraction at batch time without densifying full matrix.

Code excerpt:

```python
X = hstack([
    overview_m,
    csr_matrix(genres_m),
    csr_matrix(keywords_m),
    csr_matrix(adult_m),
    csr_matrix(numeric_m)
], format='csr')
```

## Graph Loading and Normalization

```python
A = coo_matrix((weights, (src_idx, dst_idx)), shape=(X.shape[0], X.shape[0])).tocsr()
if CFG['symmetrize_graph']:
    A = (A + A.T) * 0.5
row_sums = np.asarray(A.sum(axis=1)).reshape(-1)
row_sums[row_sums == 0] = 1.0
D_inv = csr_matrix((1/row_sums, (np.arange(len(row_sums)), np.arange(len(row_sums)))), shape=A.shape)
A_norm = D_inv.dot(A)
```

Why:

- Symmetrization ensures mutual similarity where possible, reducing directional sparsity artifacts from kNN construction.
- Row-normalization (random-walk matrix) prevents degree bias during smoothing; each node’s influence is normalized.

// ...existing code...

## Model Architecture: Shared Item Encoder + User Embeddings

```python
class ItemUserModel(nn.Module):
    def __init__(self, in_dim, emb_dim, n_users, hidden_dim=256, dropout=0.2):
        self.item_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)

    def forward(self, Xb, user_idx):
        h = self.item_net(Xb)         # item embedding
        u = self.user_emb(user_idx)   # user embedding
        b = self.user_bias(user_idx).squeeze(-1)
        logit = (h * u).sum(dim=1) + b
        return logit, h
```

Why dot product + bias:

- Factorization-style interaction (element-wise product then sum) is a standard for implicit feedback models.
- User bias absorbs tendency to rate higher/lower than mean after binarization.
- Shared item encoder learns global content representation transferable across users.

### What this model actually learns

- Item encoder (item_net): a shared feature extractor that maps high‑dimensional sparse content (TF‑IDF, genres, etc.) into a compact embedding h. Think of h as the “content coordinates” of a movie along taste dimensions (humor, drama, pace, etc.), learned from all users at once.
- User embedding (user_emb): for each user we learn a vector u in the same space. It represents how much the user likes each latent dimension from the item encoder.
- User bias (user_bias): a personal offset capturing a user’s general tendency to like/dislike more often after binarization.

The score is a compatibility measure: dot(h, u) + b. If a movie’s content dimensions align with a user’s preferences, the score goes up; otherwise it goes down.

### One model for all users: how training works

We train on all (user, movie, label) triples together with a single set of item_net weights shared across users:

- Each batch mixes many users. For every pair, we:
  - Build h = item_net(X_item)
  - Look up u, b for that specific user
  - Predict logit = dot(h, u) + b and apply BCEWithLogitsLoss to match the user’s binary label.
- Gradients flow into:
  - item_net (shared): nudged by feedback from all users, so it learns content features that are broadly predictive.
  - user_emb and user_bias (per-user): adjusted only by that user’s interactions, specializing preferences.

This is multi-task learning: every user is a tiny “task,” and the item encoder is a shared backbone. The backbone benefits from the data of everyone, which is why we can generalize even with few interactions per user.

### Why this works in practice

- Shared content backbone: items with similar content cluster in embedding space, so even users with few labels can get good signals.
- Simple, robust interaction: dot product is a smooth, well‑behaved compatibility measure that regularizes naturally (no explosion of parameters).
- Personalization with few parameters: per user we only learn u and b, which is data‑efficient and fast to train.
- Cold‑start for items: new items can be embedded immediately from their content via item_net (no historical clicks needed).

### Training objective and supervision

- Labels are per-user likes vs. not likes (thresholded by the user’s own mean rating).
- Loss: Binary cross entropy with logits over predicted probabilities.
- Optimization: Adam updates both shared (item_net) and user‑specific (embeddings, bias) parameters.
- Regularization: Dropout in item_net reduces overfitting to spurious text features.

### Generalization and limitations

- New items: supported out‑of‑the‑box (content → item_net → h).
- New users: need a few interactions to learn u and b; until then, you can default to the population mean or a small warm‑start fit.
- Limits: dot product models linear compatibility; if interactions are highly non‑linear, a deeper interaction head or attention can help (at extra cost).

In short, we train one global content encoder for everyone and small per‑user preference vectors. The encoder learns universal “what this movie is,” while user embeddings learn “what this person likes,” and the dot product ties them together in a simple, scalable way.

## Memory-Efficient Training: Lazy Dense Conversion per Batch

```python
rows = [X_sparse[int(m.item())].toarray() for m in mids]
Xb = torch.tensor(np.vstack(rows), dtype=torch.float32)
```

- Avoid loading full dense matrix into GPU memory when TF-IDF dimension is large.
- CSR row slicing is cheap; conversion only for the batch minimizes peak memory usage.

## Base Predictions vs. Correct-and-Smooth

Two phases of refinement:

1. Baseline scores per item: `sigmoid((h * u).sum + b)`.
2. Correction on labeled training items: push predictions closer to observed labels.
3. Smoothing: diffuse corrected scores across graph neighbors.

### Correction Step

```python
y_corr = y_soft.copy()
y_corr[train_idx] += alpha * (y_train - y_soft[train_idx])
```

- Anchor predictions at known labels without retraining the model.
- Linear residual update: if prediction was too low for a positive label, it is boosted proportionally.
- `alpha` controls trust in labels vs. model; avoids overfitting by partial adjustment.

### Smoothing Step

```python
y_sm = y_corr.copy()
for _ in range(T):
    y_neigh = A_norm.dot(y_sm)
    y_sm = (1-beta)*y_sm + beta*y_neigh
```

- `(1-beta)` retains self-information; `beta` injects neighbor influence.
- Row-normalized adjacency transforms scores into neighborhood averages (random-walk semantics).
- Few iterations (`T=1`) reduce oversmoothing risk while adding collaborative signal.

## Metric Collection and Fair Comparison

For each user:

- Compute baseline metrics on test subset.
- Apply C+S and recompute metrics.
- Skip users with test sets having < 2 unique labels (cannot compute F1/ROC AUC).

Why per-user reporting:

- Captures personalization rather than aggregating over heterogeneous rating behaviors.
- Enables distributional analysis of gains (e.g., which users benefit most from smoothing).

## Hyperparameters: Rationale

- `emb_dim=64`: balances representational capacity and overfitting risk for moderate dataset size.
- `hidden_dim=256`: provides non-linear expansion before bottleneck embedding.
- `dropout=0.2`: regularizes item encoder against spurious feature co-adaptations.
- `alpha=0.8`: strong but not absolute trust in observed labels.
- `beta=0.5`: equal mixing of self vs. neighborhood to avoid washing out user-specific taste signals.
- `T=1`: empirical trade-off; higher T risks homogenizing scores across densely connected subgraphs.
- `batch_size=2048`: amortizes GPU kernel launch overhead; large but feasible for sparse → dense conversion.

## Potential Improvements

1. Adaptive `alpha` based on user training size (more data → stronger correction).
2. Replace linear item encoder with GraphSAGE/GCN to integrate graph during initial training.

## Evaluation Interpretation

- Positive deltas (C+S - base) on metrics like ROC AUC indicate successful exploitation of graph context.
- If precision increases but recall drops, correction may be conservative; tune `alpha` downward.
- Flat metrics suggest graph lacks additional signal beyond content features.

## Summary

This pipeline couples a scalable content-based embedding model with a post-hoc, graph-aware refinement that is computationally cheap and improves personalization without retraining. Each component (personal mean threshold, residual correction, limited smoothing) was chosen to balance accuracy gains, interpretability, and runtime efficiency.
