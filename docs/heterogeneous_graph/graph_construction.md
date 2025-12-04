# Constructing a Heterogeneous Recommendation Graph

This file describes the process of constructing a heterogeneous graph for a movie recommendation system, which serves as the foundation for Graph Neural Network (GNN) training.

## Graph Structure Overview

The graph is designed as a heterogeneous network with five distinct node types and multiple relationship types, capturing the complex interactions in a movie recommendation ecosystem:

```
# Node Types:
# 1. Users - Individuals who rate movies
# 2. Movies - Films in the catalog  
# 3. Actors - Cast members
# 4. Directors - Film directors
# 5. Genres - Movie categories

# Edge Types (Relationships):
# - (user) --[rates]--> (movie)            # User-movie interactions
# - (movie) --[has_genre]--> (genre)       # Movie categorization
# - (movie) --[has_director]--> (director) # Director relationships
# - (movie) --[has_actor]--> (actor)       # Cast relationships
```

## Data Loading and Preparation

The graph is constructed from multiple CSV files obtained after the dataset preprocessing `[HGDataPreparation.ipynb]`, each representing either node features or edge relationships:

```python
# Load node features and edge lists
df_user_stats = pd.read_csv(f'{path}/user_stats.csv')          # User features
df_movie_features = pd.read_csv(f'{path}/movie_features.csv')  # Movie features
df_actors = pd.read_csv(f'{path}/actors.csv')                  # Actor features
df_directors = pd.read_csv(f'{path}/directors.csv')            # Director features
df_genres = pd.read_csv(f'{path}/genres.csv')                  # Genre features

# Load edge relationships
df_user_movie = pd.read_csv(f'{path}/user_movie_edges.csv')    # Ratings
df_movie_genre = pd.read_csv(f'{path}/movie_genre_edges.csv')  # Genre assignments
df_movie_actor = pd.read_csv(f'{path}/movie_actor_edges.csv')  # Cast information
df_movie_director = pd.read_csv(f'{path}/movie_director_edges.csv') # Director credits
```

## Feature Engineering and Normalization

Before graph construction, numerical features are standardized and categorical variables are one-hot encoded.

## ID Remapping for Graph Construction

```python
# Create mapping dictionaries for each node type
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(df_movie_features['movie_id'].unique())}
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(df_user_stats['user_id'].unique())}
director_id_to_idx = {director_id: idx for idx, director_id in enumerate(df_directors['director_id'].unique())}
actor_id_to_idx = {actor_id: idx for idx, actor_id in enumerate(df_actors['actor_id'].unique())}
genre_id_to_idx = {genre_id: idx for idx, genre_id in enumerate(df_genres['genre_id'].unique())}

mappings = {
    'user': user_id_to_idx,
    'movie': movie_id_to_idx,
    'actor': actor_id_to_idx,
    'director': director_id_to_idx,
    'genre': genre_id_to_idx
}
```

## Graph Initialization with PyG

All the nodes and the corresponding edges were translated into one heterogeneous graph.
