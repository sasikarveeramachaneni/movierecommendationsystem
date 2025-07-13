import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("movies.csv")
    df = df.fillna("")
    df['combined_features'] = df['genres'] + " " + df['keywords'] + " " + df['tagline'] + " " + df['cast'] + " " + df['director']
    return df

# Recommend function
def recommend_movies(movie_name, df, similarity):
    list_of_titles = df['title'].tolist()
    match = difflib.get_close_matches(movie_name, list_of_titles)
    if not match:
        return []
    close_match = match[0]
    index = df[df.title == close_match]['index'].values[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]].title for i in sorted_scores[1:31]]
    return recommended

# UI starts here
df = load_data()

# Vectorize features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df['combined_features'])
similarity = cosine_similarity(feature_vectors)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like, and we'll recommend similar ones!")

movie_list = sorted(df['title'].tolist())
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    results = recommend_movies(selected_movie, df, similarity)
    if results:
        st.subheader("Movies you might like:")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")
    else:
        st.write("No similar movies found. Please try another title.")
