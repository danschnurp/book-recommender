import os
import re

import pandas as pd
import numpy as np

from typing import List, Tuple
import faiss
from faiss import write_index, read_index
import gradio as gr
from fuzzywuzzy import process
from pandas import DataFrame
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel

# Global variables to store loaded data
dataset = None
faiss_index = None
normalized_data = None
book_titles = None


def is_valid_isbn(isbn):
    pattern = r'^(?:(?:978|979)\d{10}|\d{9}[0-9X])$'
    return bool(re.match(pattern, isbn))



def load_data(ratings_path, books_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, encoding='cp1251', sep=';', on_bad_lines='skip')
    ratings = ratings[ratings['Book-Rating'] != 0]

    books = pd.read_csv(books_path, encoding='cp1251', sep=';', on_bad_lines='skip')

    return ratings, books


def preprocess_data(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.merge(ratings, books, on=['ISBN'])
    return dataset.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)


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
    if os.path.exists("books.index"):
        return read_index("books.index")

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
                               target_book, ) -> pd.DataFrame:
    print(target_book, type(target_book))
    emb = create_embedding([target_book[0]])
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



def load_and_prepare_data():
    global dataset, faiss_index, normalized_data, book_titles, ratings_by_isbn

    # Download data files from Hugging Face
    ratings = "BX-Book-Ratings.csv"
    books = "BX-Books.csv"
    ratings, books = load_data(ratings, books)
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

    book_titles = dataset["Book-Title"]


def recommend_books(target_book: str):
    num_recommendations: int = 15
    global dataset, faiss_index, normalized_data, book_titles, ratings_by_isbn

    if dataset is None or faiss_index is None or normalized_data is None or book_titles is None:
        load_and_prepare_data()
        dataset['ISBN'] = dataset['ISBN'].str.strip()
        print("Before dropping duplicates:", len(dataset))
        dataset = dataset.drop_duplicates(subset=['ISBN'])
        print("After dropping duplicates:", len(dataset))

    target_book = target_book.lower()
    # Fuzzy match the input to the closest book title
    closest_match = process.extractOne(target_book, book_titles)


    correlations = compute_correlations_faiss(faiss_index, book_titles, closest_match)

    recommendations = correlations[correlations['book'] != target_book]

    # Create a mask of unique ISBNs
    unique_mask = dataset.duplicated(subset=['ISBN'], keep='first') == False

    # Apply the mask
    dataset = dataset[unique_mask]

    recommendations = recommendations.head(num_recommendations)

    dups = []
    result_df = pd.DataFrame([
        {
            "Title": dataset.loc[dataset['Book-Title'] == row['book'], 'Book-Title'].values[0],
            "Author": dataset.loc[dataset['Book-Title'] == row['book'], 'Book-Author'].values[0],
            "Year": dataset.loc[dataset['Book-Title'] == row['book'], 'Year-Of-Publication'].values[0],
            "Publisher": dataset.loc[dataset['Book-Title'] == row['book'], 'Publisher'].values[0],
            "ISBN": dataset.loc[dataset['Book-Title'] == row['book'], 'ISBN'].values[0],
            "Rating": ratings_by_isbn.loc[
                ratings_by_isbn['ISBN'] == dataset.loc[dataset['Book-Title'] == row['book'], 'ISBN'].values[
                    0], 'Book-Rating'].values[0],
            "none":  dups.append(dataset.loc[dataset['Book-Title'] == row['book'], 'ISBN'].values[0])
        }
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1)
        if dataset.loc[dataset['Book-Title'] == row['book'], 'ISBN'].values[0] not in dups
    ])

    return result_df


# Create Gradio interface
iface = gr.Interface(
    fn=recommend_books,
    inputs=[
        gr.Textbox(label="Enter a book title"),
    ],
    outputs=[
        gr.Dataframe(
            headers=["Title", "Author", "Year", "Publisher", "ISBN", "Rating"],
            type="pandas",

        )
    ],
    title="Book Recommender",
    description="Enter a book title to get recommendations based on user ratings and book similarities."
)

# Launch the app
iface.launch()