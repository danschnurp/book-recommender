import os
import pandas as pd
import numpy as np

from typing import List, Tuple
import faiss
from faiss import write_index, read_index
import gradio as gr
from fuzzywuzzy import process

# Global variables to store loaded data
dataset = None
faiss_index = None
normalized_data = None
book_titles = None


def load_data(ratings_path: str, books_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';')
    ratings = ratings[ratings['Book-Rating'] != 0]
    books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip')
    return ratings, books


def preprocess_data(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.merge(ratings, books, on=['ISBN'])
    return dataset.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)


def get_books_to_compare(data: pd.DataFrame, min_ratings: int = 8) -> List[str]:
    book_ratings = data.groupby('Book-Title')['User-ID'].count()
    return book_ratings[book_ratings >= min_ratings].index.tolist()


def prepare_correlation_dataset(data: pd.DataFrame, books_to_compare: List[str]) -> pd.DataFrame:
    ratings_data = data.loc[data['Book-Title'].isin(books_to_compare), ['User-ID', 'Book-Rating', 'Book-Title']]
    ratings_mean = ratings_data.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean().reset_index()
    return ratings_mean.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)


def build_faiss_index(data: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    transposed_data = data.T.values
    normalized_data = transposed_data / np.linalg.norm(transposed_data, axis=1)[:, np.newaxis]

    index_file = "books.index"
    if os.path.exists(index_file):
        return read_index(index_file), normalized_data

    dimension = normalized_data.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_data.astype('float32'))
    write_index(index, index_file)
    return index, normalized_data


def compute_correlations_faiss(index: faiss.IndexFlatIP, data: np.ndarray, book_titles: List[str],
                               target_book: str) -> pd.DataFrame:
    target_index = book_titles.index(target_book)
    target_vector = data[target_index].reshape(1, -1)
    k = len(book_titles)
    similarities, I = index.search(target_vector.astype('float32'), k)
    avg_ratings = np.mean(data, axis=1)
    corr_df = pd.DataFrame({
        'book': [book_titles[i] for i in I[0]],
        'corr': similarities[0],
        'avg_rating': avg_ratings[I[0]]
    })
    return corr_df.sort_values('corr', ascending=False)


def load_and_prepare_data():
    global dataset, faiss_index, normalized_data, book_titles

    # Download data files from Hugging Face
    ratings_file = "BX-Book-Ratings.csv"
    books_file = "BX-Books.csv"

    ratings, books = load_data(ratings_file, books_file)
    dataset = preprocess_data(ratings, books)
    books_to_compare = get_books_to_compare(dataset)
    correlation_dataset = prepare_correlation_dataset(dataset, books_to_compare)
    faiss_index, normalized_data = build_faiss_index(correlation_dataset)
    book_titles = correlation_dataset.columns.tolist()


def recommend_books(target_book: str, num_recommendations: int = 10) -> str:
    global dataset, faiss_index, normalized_data, book_titles

    if dataset is None or faiss_index is None or normalized_data is None or book_titles is None:
        load_and_prepare_data()

    target_book = target_book.lower()
    # Fuzzy match the input to the closest book title
    closest_match, score = process.extractOne(target_book, book_titles)

    if score < 50:  # You can adjust this threshold
        return f"No close match found for '{target_book}'. Please try a different title."

    if closest_match != target_book:
        result = f"Closest match: '{closest_match}' (similarity: {score}%)\n\n"
    else:
        result = ""

    correlations = compute_correlations_faiss(faiss_index, normalized_data, book_titles, closest_match)

    recommendations = correlations[correlations['book'] != target_book].head(num_recommendations)

    result = f"Top {num_recommendations} recommendations for '{target_book}':\n\n"
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        result += f"{i}. {row['book']} (Correlation: {row['corr']:.2f})\n"

    return result


# Create Gradio interface
iface = gr.Interface(
    fn=recommend_books,
    inputs=[
        gr.Textbox(label="Enter a book title"),
        gr.Slider(minimum=1, maximum=20, step=1, label="Number of recommendations", value=10)
    ],
    outputs=gr.Textbox(label="Recommendations"),
    title="Book Recommender",
    description="Enter a book title to get recommendations based on user ratings and book similarities."
)

# Launch the app
iface.launch()