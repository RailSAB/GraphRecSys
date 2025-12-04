# GNN-Based Movie Preference Classification with PyG: Methodology, Design Choices, and Code

This md explains the notebook `gnn_classifier.ipynb`, which trains Graph Neural Networks (GCN and GraphSAGE from PyTorch Geometric) to predict per-user like/dislike labels for movies on a precomputed movie similarity graph. We emphasize the rationale behind design decisions and include code excerpts for clarity.

## Objectives

- Build node features for movies from content signals (text, genres, keywords, numeric).
- Construct a graph from precomputed edges with weights and create a PyG `Data` object.
- For each eligible user, create binary labels relative to the user’s rating scale.
- Train GCN and GraphSAGE models to produce per-user predictions on a shared graph.
- Sample a fixed subset of users (e.g., 500) that preserves the original distribution of interaction counts to reduce runtime while keeping representativeness.

Why GNNs here:

- GNNs enable label information to propagate across similar movies through the graph structure.
- Unlike global smoothing, the model learns functions of node features and graph neighborhoods end-to-end.

## Configuration and Environment

Key configuration stored in `CFG` controls graph path, model sizes, training schedule, and sampling.

```python
CFG = {
    'graph_path': 'processed_data/movie_similarity_graph.csv',
    'symmetrize_graph': True,
    'add_self_loops': False,
    'tfidf_max_features': 5000,
    'svd_dim': 256,
    'hidden_dim': 256,
    'dropout': 0.2,
    'epochs': 5,
    'patience': 2,
    'lr': 1e-3,
    'min_interactions': 10,
    'test_size': 0.2,
    'random_state': 42,
    'sample_user_count': 500,
    'user_sample_seed': 123,
    'strat_bins_max': 20,
    'results_dir': 'results/gnn',
}
```

Why SVD and small epochs:

- Running full-batch GNNs on dense features is expensive. We reduce the sparse feature matrix with TruncatedSVD to a compact dense embedding.
- Lower epoch counts with early stopping (patience) keep runtime practical.

## Data Loading and Graph Construction

We load processed movies, ratings, and edges, then filter everything to a consistent set of movie IDs.

```python
movies = pd.read_csv('processed_data/movies_processed.csv')
ratings = pd.read_csv('processed_data/ratings_with_tmdb.csv')
edges_raw = pd.read_csv(CFG['graph_path'])

movie_ids = sorted(movies['id'].dropna().unique().tolist())
id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
ratings = ratings[ratings['tmdbId'].isin(movie_ids)].copy()

src_col = [c for c in edges_raw.columns if c.lower()=='source'][0]
dst_col = [c for c in edges_raw.columns if c.lower()=='target'][0]
w_col   = [c for c in edges_raw.columns if c.lower()=='weight'][0]

edges_df = edges_raw[edges_raw[src_col].isin(movie_ids) & edges_raw[dst_col].isin(movie_ids)].copy()
edges_df['src_idx'] = edges_df[src_col].map(id_to_idx)
edges_df['dst_idx'] = edges_df[dst_col].map(id_to_idx)

edge_index_list = []
edge_weight_list = []
for r in edges_df.itertuples():
    edge_index_list.append([r.src_idx, r.dst_idx])
    edge_weight_list.append(float(getattr(r, w_col)))

if CFG['symmetrize_graph']:
    rev_edges = []
    rev_weights = []
    for (s,d), w in zip(edge_index_list, edge_weight_list):
        rev_edges.append([d,s])
        rev_weights.append(w)
    edge_index_list.extend(rev_edges)
    edge_weight_list.extend(rev_weights)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
```

Why symmetrize:

- Many kNN constructions are directed; symmetrization restores mutual relationships and stabilizes message passing layers which typically assume undirected edges or benefit from them.

## Feature Engineering and Dimensionality Reduction

We replicate content features to other notebooks: TF‑IDF on overview text, one-hot for multi-label genres and keywords, adult flag, and scaled numeric columns. Then we reduce with SVD and standardize to get a dense `x` for GNN.

```python
# Build sparse features (TF-IDF + OHE + numeric)
X_sparse = hstack([
    overview_m,
    csr_matrix(genres_m),
    csr_matrix(keywords_m),
    csr_matrix(adult_m),
    csr_matrix(numeric_m)
], format='csr')

# Reduce and scale
svd = TruncatedSVD(n_components=CFG['svd_dim'], random_state=CFG['random_state'])
X_dense = svd.fit_transform(X_sparse)
scaler_dense = StandardScaler()
X_dense = scaler_dense.fit_transform(X_dense).astype(np.float32)
x = torch.tensor(X_dense, dtype=torch.float)
```

Why SVD before GNN:

- GCN/SAGE are full-batch in this notebook; reducing feature dimensionality lowers memory and compute without discarding the graph structure.

## PyG Data Object

We create a single PyG `Data` object shared across users.

```python
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
```

Why one shared graph:

- The graph encodes content similarity independent of user; per-user supervision is introduced via masks and labels.

## Per-User Labeling and Masks

Labels are binarized per user around the user’s mean rating. We add stratified sampling over users to limit runtime while preserving the interaction-count distribution.

```python
# Eligibility by min_interactions
eligible_users = []
user_interactions = {}
for user, df in ratings.groupby('userId'):
    if df.shape[0] >= CFG['min_interactions']:
        eligible_users.append(user)
        user_interactions[user] = df.copy()

# Stratified sampling by interaction count (qcut bins)
counts_df = pd.DataFrame({
    'userId': eligible_users,
    'n_interactions': [user_interactions[u].shape[0] for u in eligible_users],
})
q_bins = min(CFG['strat_bins_max'], counts_df['n_interactions'].nunique())
counts_df['bin'] = pd.qcut(counts_df['n_interactions'], q=q_bins, duplicates='drop')

bin_sizes = counts_df['bin'].value_counts().sort_index()
proportions = bin_sizes / bin_sizes.sum()
raw_alloc = proportions * CFG['sample_user_count']
alloc = raw_alloc.round().astype(int)
# rounding fix to match total
...

sampled_users = []
for b in bin_sizes.sort_index().index:
    subset = counts_df[counts_df['bin'] == b]
    k = min(alloc.loc[b], subset.shape[0])
    chosen = subset.sample(n=k, random_state=CFG['user_sample_seed'])['userId'].tolist()
    sampled_users.extend(chosen)

# Build per-user masks
def build_user_masks(user_df):
    mean_r = user_df['rating'].mean()
    user_df = user_df.assign(label=(user_df['rating'] >= mean_r).astype(int))
    train_df, test_df = train_test_split(user_df, test_size=CFG['test_size'], random_state=CFG['random_state'])
    if test_df.shape[0] < 5:
        return None
    y = torch.full((data.num_nodes,), -1.0)
    for r in train_df.itertuples():
        y[id_to_idx[r.tmdbId]] = float(r.label)
    for r in test_df.itertuples():
        y[id_to_idx[r.tmdbId]] = float(r.label)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[[id_to_idx[m] for m in train_df['tmdbId']]] = True
    test_mask[[id_to_idx[m] for m in test_df['tmdbId']]] = True
    return y.to(DEVICE), train_mask.to(DEVICE), test_mask.to(DEVICE), train_df, test_df
```

Why stratified sampling:

- Preserves the diversity of users by number of ratings, ensuring the sampled set reflects the global distribution and avoids biasing toward highly active or barely active users.
- A fixed list reused by both models guarantees a fair runtime and apples-to-apples comparison.

Why per-user mean threshold:

- Normalizes individual scales, making like/dislike consistent for each user without cross-user bias.

## Models: GCN and GraphSAGE

We define compact 2-layer variants for both models.

```python
class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x.squeeze(-1)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)
```

Why these choices:

- Two layers are sufficient for homophilic graphs; deeper models can oversmooth or overfit with limited supervision per user.
- GCN uses edge weights directly; SAGE typically ignores weights in its standard form but brings robustness via neighborhood aggregation.

## Training Loops and Metrics

For each sampled user, we train a model from scratch on the shared `Data` using only the user’s masks and labels.

```python
criterion = nn.BCEWithLogitsLoss()
for user in sampled_users:
    masks = build_user_masks(user_interactions[user])
    if masks is None:
        continue
    y, train_mask, test_mask, train_df, test_df = masks
    model = GCNClassifier(in_dim=data.x.size(-1), hidden_dim=CFG['hidden_dim'], dropout=CFG['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)
    stopper = EarlyStopping(patience=CFG['patience'])
    for epoch in range(CFG['epochs']):
        model.train(); optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward(); optimizer.step()
        stopper.step(loss.item())
        if stopper.stop:
            break
    model.eval(); logits = model(data.x, data.edge_index, data.edge_weight).detach()
    probs = torch.sigmoid(logits)
    test_probs = probs[test_mask].cpu().numpy()
    test_labels = y[test_mask].cpu().numpy().astype(int)
    # compute metrics and collect
```

Why train per user instead of multitask head:

- Keeps the protocol simple and isolates the impact of graph + features for each user.
- Avoids interference across users with highly different taste profiles.

Metrics used:

- Accuracy, F1, ROC AUC (if both classes present), Precision, Recall.


## Results and Comparisons

We save per-user results into separate CSVs for GCN and GraphSAGE and compute mean metrics and deltas.

```python
gcn_df.to_csv('results/gnn/gnn_gcn_results.csv', index=False)
sage_df.to_csv('results/gnn/gnn_graphsage_results.csv', index=False)

metrics_cols = ['accuracy','f1','roc_auc','precision','recall']
gcn_mean = gcn_df[metrics_cols].mean(numeric_only=True)
sage_mean = sage_df[metrics_cols].mean(numeric_only=True)
comp = pd.DataFrame({'GCN': gcn_mean, 'GraphSAGE': sage_mean, 'Delta(SAGE-GCN)': sage_mean - gcn_mean})
```

## Scalability and Runtime

- Dimensionality reduction (SVD) makes full-batch GNN training feasible.
- Stratified sampling of users dramatically reduces runtime while maintaining representativeness.
- Early stopping avoids wasted epochs when loss plateaus.

Potential accelerations:

- Use mini-batch techniques (NeighborSampler) for very large graphs.
- Cache `x` on GPU if memory allows; otherwise keep on CPU and move model to CPU as needed.
- Precompute normalized/symmetric edge structures.

## When to Prefer GCN vs. GraphSAGE

- GCN: leverages explicit edge weights and often works well on homophilic graphs built from cosine/UMAP similarities.
- GraphSAGE: more robust to degree variance and can generalize better with larger hidden sizes or deeper stacks when more supervision is available.

## Summary

The notebook builds a principled GNN pipeline for personalized movie like/dislike prediction:

- Solid content-derived features reduced via SVD,
- A shared movie graph and PyG `Data` object,
- Per-user binarization and stratified sampling,
- Two competitive GNNs (GCN and GraphSAGE),
- Consistent evaluation and artifact saving.

These choices balance accuracy, fairness of comparison, and practical runtime on typical hardware.
