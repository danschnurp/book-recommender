import os.path
import time

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import faiss
from faiss import write_index, read_index


def load_data(ratings_path: Path, books_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';')
    ratings = ratings[ratings['Book-Rating'] != 0]

    books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip')

    return ratings, books


def preprocess_data(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.merge(ratings, books, on=['ISBN'])
    return dataset.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)


def get_tolkien_readers(data: pd.DataFrame) -> np.ndarray:
    tolkien_mask = (data['Book-Title'] == 'the fellowship of the ring (the lord of the rings, part 1)') & \
                   (data['Book-Author'].str.contains("tolkien"))
    return data.loc[tolkien_mask, 'User-ID'].unique()


def get_books_to_compare(data: pd.DataFrame, min_ratings: int = 8) -> List[str]:
    book_ratings = data.groupby('Book-Title')['User-ID'].count()
    return book_ratings[book_ratings >= min_ratings].index.tolist()


def prepare_correlation_dataset(data: pd.DataFrame, books_to_compare: List[str]) -> pd.DataFrame:
    ratings_data = data.loc[data['Book-Title'].isin(books_to_compare), ['User-ID', 'Book-Rating', 'Book-Title']]
    ratings_mean = ratings_data.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean().reset_index()
    return ratings_mean.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)


def build_faiss_index(data: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    # Transpose the data so each column represents a book
    transposed_data = data.T.values

    # Normalize the data
    normalized_data = transposed_data / np.linalg.norm(transposed_data, axis=1)[:, np.newaxis]

    if os.path.exists("data/books.index"):
        return read_index("data/books.index"), normalized_data

    # Create a Faiss index
    dimension = normalized_data.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to the index
    index.add(normalized_data.astype('float32'))

    write_index(index, "data/books.index")


    return index, normalized_data


def compute_correlations_faiss(index: faiss.IndexFlatIP, data: np.ndarray, book_titles: List[str],
                               target_book: str) -> pd.DataFrame:
    target_index = book_titles.index(target_book)

    target_vector = data[target_index].reshape(1, -1)

    # Perform the search
    k = len(book_titles)  # Search for all books
    similarities, I = index.search(target_vector.astype('float32'), k)

    # Compute average ratings
    avg_ratings = np.mean(data, axis=1)

    corr_df = pd.DataFrame({
        'book': [book_titles[i] for i in I[0]],
        'corr': similarities[0],
        'avg_rating': avg_ratings[I[0]]
    })

    return corr_df.sort_values('corr', ascending=False)


def main(target="Harry Potter and the Sorcerer\'s Stone (Book 1)"):
    data_dir = Path('data')
    ratings, books = load_data(data_dir / 'BX-Book-Ratings.csv', data_dir / 'BX-Books.csv')

    dataset = preprocess_data(ratings, books)
    tolkien_readers = get_tolkien_readers(dataset)

    books_to_compare = get_books_to_compare(dataset)

    correlation_dataset = prepare_correlation_dataset(dataset, books_to_compare)

    # Build Faiss index
    faiss_index, normalized_data = build_faiss_index(correlation_dataset)

    target_book = target.lower()
    correlations = compute_correlations_faiss(faiss_index, normalized_data, correlation_dataset.columns.tolist(),
                                              target_book)

    print(f"Top 10 correlated books for '{target_book}':")
    print(correlations.head(10))

    print("\nBottom 10 correlated books:")
    print(correlations.tail(10))


if __name__ == "__main__":
    # main(target='the fellowship of the ring (the lord of the rings, part 1)')
    t1 = time.time()
    main()
    print(time.time() - t1, "seconds")