import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

books_df = pd.read_csv("books.csv")
ratings_df = pd.read_csv("ratings.csv")

user_ratings_count = ratings_df['user_id'].value_counts()
ratings_df = ratings_df[ratings_df['user_id'].isin(user_ratings_count[user_ratings_count >= 200].index)]

book_ratings_count = ratings_df['book_id'].value_counts()
ratings_df = ratings_df[ratings_df['book_id'].isin(book_ratings_count[book_ratings_count >= 100].index)]

ratings_pivot = ratings_df.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
ratings_matrix = csr_matrix(ratings_pivot.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(ratings_matrix)

def get_recommends(book_title):
    book_id = books_df[books_df['title'] == book_title]['book_id'].iloc[0]
    distances, indices = knn_model.kneighbors(ratings_pivot.loc[book_id].values.reshape(1, -1), n_neighbors=6)
    recommends = []
    for i in range(1, len(distances.flatten())):
        recommended_book_id = ratings_pivot.index[indices.flatten()[i]]
        recommended_book_title = books_df[books_df['book_id'] == recommended_book_id]['title'].iloc[0]
        recommends.append([recommended_book_title, distances.flatten()[i]])
    return [book_title, recommends]

get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
