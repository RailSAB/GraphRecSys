# Building a Movie Similarity Graph: Methodology and Code Walkthrough

This md explains how the notebook `graph_construction.ipynb` constructs a weighted movie similarity graph from raw TMDB/Movielens-like metadata. It covers the data sources, preprocessing steps, feature engineering, graph building with kNN, UMAP-style fuzzy edge weighting, truncation, and basic graph statistics. Code excerpts are provided to illustrate the approach.

## Goals

- Transform raw movie metadata into machine-usable feature vectors.
- Build a k-nearest neighbor (kNN) similarity graph between movies.
- Provide two edge-weighting schemes:
  - Simple cosine-derived weights.
  - UMAP-style fuzzy weights with symmetrization.
- Export edge lists for downstream models and analysis.

## Data Sources

- `data/movies_metadata.csv`: core movie info: `id`, `title`, `adult`, `genres`, `overview`, `popularity`, `vote_average`, `vote_count`.
- `data/keywords.csv`: movie-level keywords.
- `data/links.csv`: mapping between `movieId` (MovieLens) and `tmdbId` (TMDB id). Used to align identifiers.

Relevant excerpt:

```python
movies_df = pd.read_csv('data/movies_metadata.csv')
keywords_df = pd.read_csv('data/keywords.csv')
links_df = pd.read_csv('data/links.csv')

movies = movies_df[['id', 'title', 'adult', 'genres', 'overview', 'popularity', 'vote_average', 'vote_count']].copy()
links = links_df[['movieId', 'tmdbId']].copy()
links['movieId'] = links['movieId'].astype(np.int64)
links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').astype('Int64')

movies['id'] = pd.to_numeric(movies['id'], errors='coerce').astype('Int64')
movies = movies.merge(links, left_on='id', right_on='tmdbId', how='left')
movies.drop(columns=['tmdbId'], inplace=True)
```

## Preprocessing and Normalization

Parse JSON-like columns, handle missing values, and enforce consistent types. Dropped rows missing essential fields like `id` or `title`.

```python
# genres: convert JSON-like string to list of names
movies['genres'] = (
    movies['genres']
    .fillna('[]')
    .apply(eval)
    .apply(lambda x: [y['name'] for y in x] if isinstance(x, list) else [])
)

# keywords: join as list of names
keywords_df['keywords'] = keywords_df['keywords'].fillna('[]').apply(eval)
movies = movies.merge(keywords_df[['id', 'keywords']], on='id', how='left')
movies['keywords'] = movies['keywords'].apply(lambda x: [y['name'] for y in x] if isinstance(x, list) else [])

# drop rows missing id/title
movies.dropna(subset=['id', 'title'], inplace=True)
```

## Feature Engineering

Builded a mixed representation combining text, categorical multi-labels, and numeric signals:

- Text: `overview` via TF-IDF (max 5000 features).
- Categorical: one-hot encodings for `genres` and `keywords` via `MultiLabelBinarizer`.
- Binary flag: `adult` mapped to {0,1}.
- Numeric: `popularity`, `vote_average`, `vote_count` with `log1p(vote_count)` and `StandardScaler`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.sparse import hstack

# 1) Overview TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# 2) Genres / 3) Keywords
mlb_genres = MultiLabelBinarizer(); genres_m = mlb_genres.fit_transform(movies['genres'])
mlb_keywords = MultiLabelBinarizer(); keywords_m = mlb_keywords.fit_transform(movies['keywords'])

# 4) Adult flag
adult_mask = movies['adult'].fillna('False').map({'True': 1, 'False': 0}).values.reshape(-1, 1)

# 5) Numeric
numeric_features = movies[['popularity', 'vote_average', 'vote_count']].fillna(0)
numeric_features['popularity'] = pd.to_numeric(numeric_features['popularity'], errors='coerce').fillna(0)
numeric_features['vote_count'] = np.log1p(numeric_features['vote_count'])
scaler = StandardScaler(); numeric_m = scaler.fit_transform(numeric_features)

# Combined sparse feature matrix
X = hstack([tfidf_matrix, genres_m, keywords_m, adult_mask, numeric_m])
```

Why: This representation captures semantic content (TF-IDF), content taxonomy (genres, keywords), and popularity/quality signals.

## kNN Graph Construction (Cosine Metric)

Constructed a kNN graph using cosine distance on the combined sparse features. For each movie, we connect to its top-k neighbors (excluding self).

```python
from sklearn.neighbors import NearestNeighbors
k = 20
nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X)
distances, neighbors = nn.kneighbors(X)

edge_list = []
for i in range(len(movies)):
    movie_i_id = movies.iloc[i]['id']
    for idx, j in enumerate(neighbors[i]):
        if i == j:
            continue
        movie_j_id = movies.iloc[j]['id']
        dist = distances[i][idx]
        weight = 1 - dist  # cosine similarity
        edge_list.append((movie_i_id, movie_j_id, weight))

edges_df = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
edges_df.to_csv('processed_data/movie_similarity_graph.csv', index=False)
```

Notes:

- `metric='cosine'` works well with high-dimensional sparse TF-IDF.
- The resulting graph is directed by construction; you can later symmetrize if needed.

## UMAP-Style Fuzzy Weights and Symmetrization

We convert raw kNN distances into soft, topology-aware affinities using UMAP’s per-node normalization. The idea is to adapt each node’s neighborhood to local density so dense regions don’t get over-connected and sparse regions aren’t neglected.

- Local connectivity (rho): For each node i, set `rho_i` to the smallest non‑zero neighbor distance. Subtracting `rho_i` makes the closest valid neighbor have near‑zero effective distance, yielding a weight close to 1 and guaranteeing at least one strong local edge.
- Per-node scale (sigma): Choose `sigma_i` so that the total affinity “mass” around i matches a conservative target:
  `sum_j exp(-(d_ij - rho_i)/sigma_i) ≈ log2(k)`.
  We find `sigma_i` by a fast binary search.

Why log2(k)?
- `log2(k)` is the base‑2 logarithm of k (e.g., k=16 → `log2(k)=4`). It grows sublinearly, acting as a compact, stable “effective neighborhood size.”
- Targeting `log2(k)` prevents too many neighbors from getting high weight (over-smoothing) while preserving a handful of strong links, comparable across regions of different density.
- It plays a role similar to t‑SNE’s perplexity, without requiring a full probability normalization.

Steps
1) Compute `rho_i` as the smallest non‑zero distance among neighbors of node i.  
2) Find `sigma_i` via binary search so `sum_j exp(-(d_ij - rho_i)/sigma_i) ≈ log2(k)`.  
3) Define directional fuzzy weight `w_ij = exp(-(d_ij - rho_i)/sigma_i)`.  
4) Symmetrize with probabilistic union `w = 1 - (1 - w_ij) * (1 - w_ji)` so an undirected edge is strong if either direction is strong (and even stronger if both are).

This UMAP-style weighting yields smoother, locally balanced neighborhoods and more informative edge weights than raw cosine similarities.

```python
n = X.shape[0]
rhos = np.zeros(n)
sigmas = np.zeros(n)

for i in range(n):
    d = distances[i]
    d_nonzero = d[d > 0]
    rhos[i] = np.min(d_nonzero) if len(d_nonzero) else 0
    target = np.log2(k)
    low, high = 1e-3, 10
    for _ in range(50):
        mid = (low + high) / 2
        sum_w = np.sum(np.exp(-(d - rhos[i]) / mid))
        if abs(sum_w - target) < 1e-3:
            break
        if sum_w > target:
            high = mid
        else:
            low = mid
    sigmas[i] = mid

# directional weights
rows = []
for i in range(n):
    movie_i = movies.iloc[i]['id']
    for idx, j in enumerate(neighbors[i]):
        if i == j:
            continue
        movie_j = movies.iloc[j]['id']
        d = distances[i][idx]
        w = np.exp(-(d - rhos[i]) / sigmas[i])
        rows.append((movie_i, movie_j, w))

df = pd.DataFrame(rows, columns=['source', 'target', 'weight'])

# symmetric union
df_rev = df.rename(columns={'source':'target', 'target':'source', 'weight':'w_rev'})
dfm = df.merge(df_rev, on=['source','target'], how='outer').fillna(0)
dfm['weight'] = 1 - (1 - dfm['weight']) * (1 - dfm['w_rev'])

edges_final = dfm[['source', 'target', 'weight']]
edges_final = edges_final[edges_final['source'] != edges_final['target']]

edges_final.to_csv('processed_data/umap_movie_graph.csv', index=False)
```

Why: UMAP-style weighting better reflects local neighborhood structure and yields smoother, more informative edge weights than raw cosine similarities.

## Truncation and Processed Output

Dropped zero-weight edges and save a compact version under `processed_data/` for downstream training.

```python
edges_final = edges_final[edges_final['weight'] > 0]
edges_final.to_csv('processed_data/umap_movie_graph_truncated.csv', index=False)
```

Downstream notebooks (e.g., GNN classifiers) can consume either:

- `processed_data/movie_similarity_graph.csv` or
- `processed_data/umap_movie_graph_truncated.csv` for UMAP-style affinities.

## Reproducibility and Configuration

Key parameters to consider:

- `k` (neighbors per node): 10–50 typical; higher k increases density and computation.
- `max_features` in TF-IDF: balances vocabulary richness vs. memory.
- Numeric scaling: always standardize; keep `log1p` on heavy-tailed counts.
- UMAP target: `log2(k)` is standard from UMAP literature; can be tuned.

Seed control: scikit-learn’s `NearestNeighbors` is deterministic given inputs; randomness primarily arises from any sampling and from floating-point non-determinism if run across different BLAS backends.

## Outputs Summary

- `processed_data/movie_similarity_graph.csv`: cosine-based kNN edges.
- `processed_data/umap_movie_graph_truncated.csv`: positive-weight subset suitable for models.
- `processed_data/movies_processed.csv`: cleaned movie table for reuse.
