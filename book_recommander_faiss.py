import os.path
import time

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import faiss
from faiss import write_index, read_index
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
import torch
from tqdm import tqdm

from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel


def is_valid_isbn(isbn):
    pattern = r'^(?:(?:978|979)\d{10}|\d{9}[0-9X])$'
    return bool(re.match(pattern, isbn))


def load_data(ratings_path: Path, books_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';', on_bad_lines='skip')
    ratings = ratings[ratings['Book-Rating'] != 0]

    books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip')

    return ratings, books


def preprocess_data(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.merge(ratings, books, on=['ISBN'])
    return dataset.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)


def prepare_correlation_dataset(data: pd.DataFrame, books_to_compare: List[str]) -> pd.DataFrame:
    ratings_data = data.loc[data['Book-Title'].isin(books_to_compare), ['Book-Title']]
    return ratings_data.pivot(index='ISBN', columns='Book-Title', values='Book-Title').fillna(0)


def create_embedding(dataset):
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("creating tokens")
    tokens = [tokenizer(i, padding="max_length", truncation=True, max_length=10, return_tensors='pt')
              for i in dataset]
    print("\ncreating embedding\n")
    emb = []
    for i in tqdm(tokens):
        emb.append(model(**i,)["last_hidden_state"].detach().numpy().squeeze().reshape(-1))
    # Normalize the data
    normalized_data = emb / np.linalg.norm(emb)
    return normalized_data


def build_faiss_index(dataset: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if os.path.exists("data/books.index"):
        return read_index("data/books.index")

    dataset["embedding"] = create_embedding(dataset["Book-Title"])
    print("creating index")
    normalized_data = dataset["embedding"]
    # Create a Faiss index
    dimension = normalized_data.shape[-1]
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to the index
    index.add(normalized_data.astype('float16'))

    write_index(index, "data/books.index")

    return index


def compute_correlations_faiss(index: faiss.IndexFlatIP, book_titles: List[str],
                               target_book: str, ) -> pd.DataFrame:
    emb = create_embedding([target_book])
    # target_vector = book_titles.index(emb)


    # Perform the search
    k = len(book_titles)  # Search for all books
    similarities, I = index.search(emb.astype('float16'), k)

    # # Reduce database and query vectors to 2D for visualization
    # pca = PCA(n_components=2)
    # reduced_db = pca.fit_transform(data)
    # reduced_query = pca.transform(target_vector)
    #
    # # Scatter plot
    # plt.scatter(reduced_db[:, 0], reduced_db[:, 1], label='Database Vectors', alpha=0.5)
    # plt.scatter(reduced_query[:, 0], reduced_query[:, 1], label='Query Vectors', marker='X', color='red')
    # plt.legend()
    # plt.title("PCA Projection of IndexFlatIP Vectors")
    # plt.show()



    corr_df = pd.DataFrame({
        'book': [book_titles[i] for i in I[0]],
        'corr': similarities[0]
    })
    return corr_df.sort_values('corr', ascending=False)


def main(target="Harry Potter and the Sorcerer\'s Stone (Book 1)"):
    data_dir = Path('data')
    ratings, books = load_data(data_dir / 'BX-Book-Ratings.csv', data_dir / 'BX-Books.csv')

    dataset = preprocess_data(ratings, books)
    ratings = ratings[ratings['ISBN'].apply(is_valid_isbn)]
    dataset = dataset[dataset['ISBN'].apply(is_valid_isbn)]

    ratings_by_isbn = ratings.drop(columns="User-ID")[ratings.drop(columns="User-ID")["Book-Rating"] > 0]
    ratings_by_isbn = ratings_by_isbn.groupby('ISBN')["Book-Rating"].mean().reset_index()
    ratings_by_isbn = ratings_by_isbn.drop_duplicates(subset=['ISBN'])
    dataset = dataset.drop(columns=["User-ID", "Book-Rating"])
    dataset = dataset[dataset['ISBN'].isin(ratings_by_isbn['ISBN'])]
    dataset = dataset.drop_duplicates(subset=['ISBN'])
    dataset = preprocess_data(dataset, ratings_by_isbn)
    # Build Faiss index
    faiss_index = build_faiss_index(dataset)

    target_book = target.lower()
    correlations = compute_correlations_faiss(faiss_index, dataset["Book-Title"],
                                              target_book)

    print(f"Top 10 correlated books for '{target_book}':")
    print(correlations.head(10))

    print("\nBottom 10 correlated books:")
    print(correlations.tail(10))


if __name__ == "__main__":
    # main(target='the fellowship of the ring (the lord of the rings, part 1)')
    t1 = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
    print(time.time() - t1, "seconds")
