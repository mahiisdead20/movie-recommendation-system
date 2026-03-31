"""
🎬 Movie Recommendation System
Hybrid: Collaborative Filtering + Content-Based Filtering
"""

import math
from collections import defaultdict

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

MOVIES = {
    1: {"title": "The Dark Knight", "genres": ["Action", "Crime", "Thriller"], "year": 2008, "rating": 9.0},
    2: {"title": "Inception", "genres": ["Action", "Sci-Fi", "Thriller"], "year": 2010, "rating": 8.8},
    3: {"title": "Interstellar", "genres": ["Adventure", "Sci-Fi", "Drama"], "year": 2014, "rating": 8.6},
    4: {"title": "The Matrix", "genres": ["Action", "Sci-Fi"], "year": 1999, "rating": 8.7},
    5: {"title": "Pulp Fiction", "genres": ["Crime", "Drama", "Thriller"], "year": 1994, "rating": 8.9},
    6: {"title": "The Shawshank Redemption", "genres": ["Drama"], "year": 1994, "rating": 9.3},
    7: {"title": "Forrest Gump", "genres": ["Drama", "Romance"], "year": 1994, "rating": 8.8},
    8: {"title": "The Godfather", "genres": ["Crime", "Drama"], "year": 1972, "rating": 9.2},
    9: {"title": "Avengers: Endgame", "genres": ["Action", "Adventure", "Sci-Fi"], "year": 2019, "rating": 8.4},
    10: {"title": "Parasite", "genres": ["Drama", "Thriller", "Comedy"], "year": 2019, "rating": 8.5},
}

USER_RATINGS = {
    "Alice": {1: 5, 2: 4, 3: 5},
    "Bob": {5: 5, 6: 5, 7: 4},
    "Charlie": {1: 4, 4: 5, 9: 5},
}

# ─────────────────────────────────────────────
# SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two users."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def genre_similarity(movie_a, movie_b):
    """Jaccard similarity between movie genres."""
    genres_a = set(MOVIES[movie_a]["genres"])
    genres_b = set(MOVIES[movie_b]["genres"])

    union = genres_a | genres_b
    if not union:
        return 0.0

    return len(genres_a & genres_b) / len(union)

# ─────────────────────────────────────────────
# RECOMMENDATION SYSTEMS
# ─────────────────────────────────────────────

def collaborative_filter(user, top_n=5):
    """Recommend movies using user-based collaborative filtering."""
    if user not in USER_RATINGS:
        return []

    user_vec = USER_RATINGS[user]
    seen = set(user_vec.keys())

    similarities = {}
    for other, vec in USER_RATINGS.items():
        if other != user:
            sim = cosine_similarity(user_vec, vec)
            if sim > 0:
                similarities[other] = sim

    scores = defaultdict(float)
    weights = defaultdict(float)

    for other, sim in similarities.items():
        for mid, rating in USER_RATINGS[other].items():
            if mid not in seen:
                scores[mid] += sim * rating
                weights[mid] += sim

    predictions = {
        mid: scores[mid] / weights[mid]
        for mid in scores if weights[mid] > 0
    }

    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]


def content_based_filter(liked_ids, top_n=5):
    """Recommend based on genre similarity."""
    if not liked_ids:
        return []

    candidates = set(MOVIES.keys()) - set(liked_ids)
    scores = {}

    for cid in candidates:
        max_sim = max(genre_similarity(cid, lid) for lid in liked_ids)
        scores[cid] = max_sim

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def hybrid_recommend(user, top_n=5):
    """Hybrid recommendation system."""
    collab = dict(collaborative_filter(user, top_n=10))
    liked = list(USER_RATINGS.get(user, {}).keys())
    content = dict(content_based_filter(liked, top_n=10))

    all_movies = set(collab.keys()) | set(content.keys())
    results = {}

    for mid in all_movies:
        c_score = collab.get(mid, 0) / 5.0
        t_score = content.get(mid, 0)
        results[mid] = 0.6 * c_score + 0.4 * t_score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    user = input("Enter user name: ")
    recs = hybrid_recommend(user)

    print("\nRecommended Movies:\n")
    for mid, score in recs:
        print(f"{MOVIES[mid]['title']} ({MOVIES[mid]['year']}) - Score: {score:.2f}")
