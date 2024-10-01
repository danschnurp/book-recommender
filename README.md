# Book Correlation Analysis using Faiss

This project implements a book correlation analysis system using collaborative filtering techniques and Faiss for efficient similarity search. The script analyzes user ratings of books to find correlations between different titles.

## Features

- Data loading and preprocessing from CSV files
- Correlation analysis of books using Faiss for efficient similarity search
- Ranking of books based on their correlation with the target book

## Technologies Used

- Python 3.11
- pandas: For data manipulation and analysis
- NumPy: For numerical operations
- Faiss: For efficient similarity search and clustering of dense vectors

## Techniques Implemented

1. **Collaborative Filtering**: The script uses a user-item matrix to represent book ratings, allowing for the discovery of similar books based on user preferences.

2. **Faiss Indexing**: Faiss is used to create an efficient index for similarity search. Specifically, we use the IndexFlatIP (Inner Product) index, which is suitable for computing correlations with normalized data.

3. **Vector Normalization**: Book rating vectors are normalized before being added to the Faiss index. This ensures that the inner product between vectors corresponds to their cosine similarity, which is equivalent to the Pearson correlation coefficient for centered data.

4. **Transposed Data Structure**: The user-item matrix is transposed so that each vector in the Faiss index represents a book rather than a user. This aligns with our goal of finding similar books.

5. **Efficient Similarity Search**: Instead of computing correlations pairwise, we use Faiss to perform a single search operation for all books, significantly improving performance for large datasets.

## How It Works

1. **Data Loading and Preprocessing**:
   - Load book ratings and book information from CSV files.
   - Merge and preprocess the data, converting text to lowercase for consistency.
   - 
2. **Preparing the Dataset**:
   - Create a user-item matrix of book ratings for Tolkien readers.
   - Filter out books with fewer than a specified number of ratings (default is 8).

3. **Building the Faiss Index**:
   - Transpose the user-item matrix so each vector represents a book.
   - Normalize the book vectors.
   - Create a Faiss IndexFlatIP and add the normalized vectors.

4. **Computing Correlations**:
   - Use Faiss to find the most similar books to the target book.
   - The similarity scores returned by Faiss represent the correlations between books.

5. **Results**:
   - Display the top 10 most correlated books and the bottom 10 least correlated books.

## Advantages of this Approach

1. **Scalability**: Faiss is designed to handle large datasets efficiently, making this approach suitable for extensive book catalogs and user bases.

2. **Performance**: Using Faiss for similarity search is significantly faster than traditional pairwise correlation computations, especially for large datasets.

3. **Memory Efficiency**: Faiss can work with datasets that might not fit entirely in memory, allowing for analysis of very large book catalogs.

4. **Flexibility**: The approach can be easily adapted to other item-based collaborative filtering tasks beyond book recommendations.

## Potential Improvements

1. **Hyperparameter Tuning**: Experiment with different Faiss index types and parameters for potential performance improvements.

2. **Time-based Analysis**: Incorporate time-based weighting of ratings to account for changing user preferences over time.

3. **Content-based Features**: Combine collaborative filtering with content-based features (e.g., book genres, authors) for a hybrid recommendation system.

4. **Evaluation Metrics**: Implement evaluation metrics such as Mean Average Precision (MAP) or Normalized Discounted Cumulative Gain (NDCG) to quantitatively assess the quality of the correlations.
