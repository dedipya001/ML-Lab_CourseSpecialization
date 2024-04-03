# Build Recommendation Systems for content based filtering and collaborative based filtering (item based) for the given dataset.
# Note: Use cosine similarity for content based filtering and centered cosine similarity for Collaborative filtering



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

movies['title'] = movies['title'].str.lower()


data = pd.merge(ratings, movies, on='movieId')

matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Content-based filtering
item_similarity = cosine_similarity(matrix.fillna(0).T)
item_similarity_df = pd.DataFrame(item_similarity, index=matrix.columns, columns=matrix.columns)

# Collaborative filtering
user_similarity = 1 - pairwise_distances(matrix.fillna(0), metric='correlation')
user_similarity_df = pd.DataFrame(user_similarity, index=matrix.index, columns=matrix.index)

def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


def get_similar_users(user_id):
    similar_score = user_similarity_df[user_id]
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


movie_name = "toy story (1995)"
user_rating = 4  #  the user rating for Toy Story is 4

similar_movies = get_similar_movies(movie_name, user_rating)
print("Content-based filtering recommendations for Toy Story: - the top 20")
print(similar_movies.head(20)) 



user_id = 1  # we're interested in recommendations for user 1

similar_users = get_similar_users(user_id)
print("\nCollaborative filtering recommendations for User 1 - the top 20:")
print(similar_users.head(20))