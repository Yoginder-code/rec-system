import streamlit as st
import pandas as pd

# Load the matrices
@st.cache_data
def load_matrices():
    user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)
    item_similarity = pd.read_csv('item_similarity.csv', index_col=0)
    item_similarity_cosine = pd.read_csv('item_similarity_cosine.csv', index_col=0)
    return user_item_matrix, item_similarity, item_similarity_cosine

user_item_matrix, item_similarity, item_similarity_cosine = load_matrices()

# Function to find the highest rated movie by a user
def get_highest_rated_movie(user_id, user_item_matrix):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    
    if user_ratings.empty:
        return None, None
    
    highest_rated_movie = user_ratings.idxmax()
    highest_rating = user_ratings.max()
    
    return highest_rated_movie, highest_rating

# Function to get top similar movies based on item similarity matrix
def get_top_similar_movies(movie_title, item_similarity_matrix, num_recommendations=5):
    if movie_title not in item_similarity_matrix.index:
        return pd.Series(dtype=float)
    
    similar_movies = item_similarity_matrix[movie_title].dropna()
    similar_movies = similar_movies[similar_movies.index != movie_title]
    
    top_recommendations = similar_movies.nlargest(num_recommendations).round(2)
    
    return top_recommendations

# Function to implement Matrix Factorization (MF) using SVD (example)
def get_mf_recommendations(user_id, user_item_matrix, num_recommendations=5):
    from sklearn.decomposition import TruncatedSVD
    import numpy as np

    user_item_matrix_filled = user_item_matrix.fillna(0)
    svd = TruncatedSVD(n_components=50)
    matrix_factors = svd.fit_transform(user_item_matrix_filled)
    
    user_factors = matrix_factors[user_id]
    item_factors = svd.components_.T
    
    scores = np.dot(item_factors, user_factors)
    sorted_indices = np.argsort(scores)[::-1]
    
    movie_ids = user_item_matrix.columns
    top_recommendations = pd.Series(scores[sorted_indices], index=movie_ids[sorted_indices]).head(num_recommendations).round(2)
    
    return top_recommendations

# Streamlit App
st.title("Movie Recommendation System")
st.write("""
Welcome to the Movie Recommendation System! This application allows you to get personalized movie recommendations based on different models. 
You can choose between Item Similarity, Cosine Similarity, and Matrix Factorization. Here's a brief explanation of each model:

- **Item Similarity**: This model finds movies that are similar to each other based on user ratings. It uses a similarity matrix where each entry indicates how similar two movies are.
- **Cosine Similarity**: This model measures the cosine of the angle between two vectors representing movies. A higher cosine similarity score indicates that the movies are more similar.
- **Matrix Factorization (MF)**: This model uses techniques like Singular Value Decomposition (SVD) to factorize the user-item matrix into lower-dimensional matrices. This helps in discovering latent features that can predict user preferences.
""")

user_id = st.number_input("Enter User ID (between 1 and 6040)", min_value=1, max_value=6040, value=1, step=1)
recommendation_method = st.selectbox("Select Recommendation Method", ("Item Similarity", "Cosine Similarity", "Matrix Factorization"))

if st.button("Get Recommendations"):
    highest_rated_movie, highest_rating = get_highest_rated_movie(user_id, user_item_matrix)
    
    if highest_rated_movie:
        st.write(f"Highest rated movie by User {user_id}: **{highest_rated_movie}** with a rating of {highest_rating}")
        
        num_recommendations = 5
        
        if recommendation_method == "Item Similarity":
            top_recommendations = get_top_similar_movies(highest_rated_movie, item_similarity, num_recommendations)
        elif recommendation_method == "Cosine Similarity":
            top_recommendations = get_top_similar_movies(highest_rated_movie, item_similarity_cosine, num_recommendations)
        elif recommendation_method == "Matrix Factorization":
            top_recommendations = get_mf_recommendations(user_id, user_item_matrix, num_recommendations)
        
        if not top_recommendations.empty:
            st.write(f"Top {num_recommendations} recommendations based on '{highest_rated_movie}':")
            for rank, (movie_title, similarity_score) in enumerate(top_recommendations.items(), start=1):
                st.write(f"{rank}. {movie_title} (Similarity Score: {similarity_score})")
        else:
            st.write(f"No recommendations found for '{highest_rated_movie}'.")
    else:
        st.write(f"User ID {user_id} has no ratings.")
