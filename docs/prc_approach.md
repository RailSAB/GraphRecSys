# Propagating Ratings on a Movie Graph: Methodology, Rationale, and Code

This md explains the methodology in `prc_approach.ipynb`: a simple yet effective per-user label propagation on a precomputed movie similarity graph. We cover the problem framing, why per-user binarization, why local propagation, data preparation, algorithm steps, practical choices, and evaluation.

## Problem Framing

We want to predict whether a user will like a movie (binary label) using a content-derived movie graph. Each user has a small set of labeled nodes (movies they rated), and we need beliefs for their unseen movies.

Why this framing:

- Cold-start resilience: content/graph features enable predictions without requiring collaborative neighbors.
- Personalization: each user runs their own propagation starting from their labels.
- Interpretability: beliefs diffuse through similar movies with weights reflecting similarity.

## Data and Graph Inputs

- Ratings with `tmdbId` and `rating` values: `processed_data/ratings_with_tmdb.csv`.
- Movie graph (UMAP-style truncated or cosine-based kNN edges): `processed_data/umap_movie_graph_truncated.csv` or `processed_data/movie_similarity_graph.csv`with columns `source, target, weight`.

Code excerpt:

```python
merged = pd.read_csv('processed_data/ratings_with_tmdb.csv')
movie_sim_graph = pd.read_csv('processed_data/umap_movie_graph_truncated.csv')
adj = movie_sim_graph.groupby('source').apply(
    lambda df: list(zip(df['target'], df['weight']))
).to_dict()
```

## Per-User Binarization

For each user, we label an item as 1 if the rating is greater or equal to the user's mean rating; otherwise 0.

Why per-user thresholding:

- Normalizes personal rating scales (some users are strict, others lenient).
- Avoids a global threshold that biases toward higher or lower raters.

```python
for user, user_df in merged.groupby('userId'):
    mean_r = user_df['rating'].mean()
    user_df['label'] = (user_df['rating'] >= mean_r).astype(int)
    if user_df.shape[0] < 10:
        continue
```

We require at least 10 interactions for stability of the mean and sufficient train/test coverage.

## Train/Test Split

We sample 80% of the user’s interactions for training and keep 20% for testing.

```python
train = user_df.sample(frac=0.8, random_state=42)
test  = user_df.drop(train.index)
if len(test) < 5:
    continue
```

We also ensure the test set is not too small; otherwise metrics can be unstable (e.g., F1 undefined).

## Initialization and Propagation

We initialize beliefs as:

- For training items: belief = true label (0 or 1).
- For test items: belief = 0.5 (uninformative prior).

Why 0.5 for unlabeled:

- Encodes uncertainty rather than assuming negative or positive.
- Prevents biasing propagation toward any class before diffusion.

Propagation iteratively updates test nodes by averaging neighbor beliefs weighted by edge weights:

```python
iterations = 50
beliefs = {}
for _, r in train.iterrows():
    beliefs[r['tmdbId']] = float(r['label'])

test_movies = test['tmdbId'].tolist()
for m in test_movies:
    beliefs[m] = 0.5

for _ in range(iterations):
    new_beliefs = beliefs.copy()
    for m in test_movies:
        neighbors = adj.get(m, [])
        if not neighbors:
            continue
        num = 0; den = 0
        for nb, w in neighbors:
            if nb in beliefs:
                num += w * beliefs[nb]
                den += w
        if den > 0:
            new_beliefs[m] = num/den
    if max(abs(new_beliefs[m]-beliefs[m]) for m in test_movies) < 1e-4:
        break
    beliefs = new_beliefs
```

Why neighbor-weighted averaging:

- It approximates harmonic functions on graphs where unlabeled nodes take the weighted average of neighbors.
- Converges to a smooth labeling consistent with labeled boundary conditions.

Why convergence check:

- Early stopping when max change < 1e-4 avoids unnecessary iterations.

Edge cases handled:

- Nodes with no neighbors (skip update) keep their prior belief (0.5), reflecting uncertainty.

## Prediction and Metrics

After convergence, we compute predicted labels and metrics for the user.

```python
y_true = test['label'].values
y_score = np.array([beliefs[m] for m in test_movies])
y_pred = (y_score >= 0.5).astype(int)

results.append({
    'userId': user,
    'accuracy': accuracy_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan,
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'train_size': len(train),
    'test_size': len(test),
})
```

## Output

The notebook aggregates per-user metrics into a DataFrame and saves it:

```python
results_df = pd.DataFrame(results)
results_df.to_csv('prc_user_based_results.csv', index=False)
```

This provides a straightforward baseline to compare against more sophisticated GNN-based models.
