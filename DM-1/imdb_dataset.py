# %%
"""
# IMDB Dataset Analysis
<img align="right" width="300" src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg">
The Internet Movie Database (IMDb) is an online database of information related to films, television series, podcasts, home videos, video games, and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews.

In this notebook, we will explore the IMDB dataset to understand the patterns and relationships between various features of movies and shows. We will use various data visualization and analysis techniques to gain insights into what factors might influence ratings and popularity.
"""

# %%
"""
**Features description**
- originalTitle: The original title of the movie/show
- rating: Rating of the movie/show (categorized into bins)
- startYear: Year when the movie/show was released/started
- endYear: Year when the show ended (for TV series)
- runtimeMinutes: Duration of the movie/episode in minutes
- awardWins: Number of awards won
- numVotes: Number of votes received
- worstRating: Minimum possible rating
- bestRating: Maximum possible rating
- totalImages: Number of images associated with the movie/show
- totalVideos: Number of videos associated with the movie/show
- totalCredits: Number of credits (cast and crew)
- criticReviewsTotal: Number of critic reviews
- titleType: Type of the title (movie, tvSeries, tvEpisode, etc.)
- awardNominationsExcludeWins: Number of award nominations (excluding wins)
- canHaveEpisodes: Whether the title can have episodes (for series)
- isRatable: Whether the title can be rated
- genres: Genres of the movie/show
"""

# %%
# Import all necessary libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import *
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import seuclidean, jaccard
import os
import sys
from scipy import interpolate

# %%
# Load the dataset and replace '\N' with NaN
df = pd.read_csv("imdb.csv", na_values='\\N')
df.head()

# %%
"""
## Part 1: Data Understanding and Exploration
"""

# %%
# Display the last few rows of the dataset
df.tail()

# %%
# Check data types of all columns
df.dtypes

# %%
# Get information about the dataset
df.info() 

# %%
"""
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html

- For numeric data, the result's index will include count, mean, std, min, max as well as percentiles
- For object data (e.g. strings), the result's index will include count, unique, top, and freq. The top is the most common value. The freq is the most common value's frequency.
"""

# %%
# Get descriptive statistics for all columns
df.describe(include="all") 

# %%
# Check the most common title type
df['titleType'].mode()

# %%
# Count of each title type
df['titleType'].value_counts()

# %%
# Count of each rating category
df['rating'].value_counts().sort_index()

# %%
"""
## Missing Values
"""

# %%
df.isnull().any()

# %%
# Check missing values and their percentage
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_values_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

missing_data = pd.concat([missing_values, missing_values_percent], axis=1, keys=['Missing Values', 'Percentage'])
missing_data = missing_data[missing_data['Missing Values'] > 0]
missing_data

# %%
"""
## Handling Missing Values

Following the professor's approach, we'll create functions to fill missing values using mean and median
"""

# %%
# Function to fill missing values with mean
def fun_fill_mean(x):
    return x.fillna(x.mean())

# %%
# Function to fill missing values with median
def fun_fill_median(x):
    return x.fillna(x.median())

# %%
"""
## Data Preprocessing
"""

# %%
"""
First, let's handle non-numeric values and missing values
"""

# %%
# Convert rating from string format to numeric if needed
# From the sample data, it looks like rating might be stored as intervals like "(7, 8]"
# Let's extract the upper bound as the numeric rating

if df['rating'].dtype == 'object':
    # Extract the upper bound from the interval notation
    try:
        df['rating_numeric'] = df['rating'].str.extract(r'\((\d+), (\d+)\]').iloc[:, 1].astype(float)
    except:
        print("Could not convert rating to numeric. Will use as is.")
        df['rating_numeric'] = df['rating']

# %%
# Handle missing values in rating_numeric
if 'rating_numeric' in df.columns and df['rating_numeric'].isnull().sum() > 0:
    # Fill missing rating values with the median
    df['rating_numeric'] = df['rating_numeric'].fillna(df['rating_numeric'].median())
    print(f"Filled {df['rating_numeric'].isnull().sum()} missing values in rating_numeric")

# %%
# Handle other non-numeric or missing values for correlation
# For startYear and endYear, convert to numeric and handle missing values
df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')

# %%
# Following the professor's approach, use grouped imputation
# Fill missing startYear values grouped by titleType
df['startYear_numeric'] = df.groupby(['titleType'])['startYear'].transform(fun_fill_median)
# If any values are still missing, fill with overall median
df['startYear_numeric'] = df['startYear_numeric'].fillna(df['startYear_numeric'].median())

# %%
# For endYear, fill based on titleType
df['endYear_numeric'] = pd.to_numeric(df['endYear'], errors='coerce')
# For series, fill missing endYear with values grouped by titleType
series_mask = df['titleType'].isin(['tvSeries', 'tvMiniSeries'])
df.loc[series_mask, 'endYear_numeric'] = df.loc[series_mask].groupby(['titleType'])['endYear_numeric'].transform(fun_fill_median)
# For movies and other types that don't typically have an end year, set it equal to the start year
non_series_mask = ~series_mask
df.loc[non_series_mask, 'endYear_numeric'] = df.loc[non_series_mask, 'startYear_numeric']

# %%
# For runtimeMinutes, fill missing values grouped by titleType
df['runtimeMinutes_numeric'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
df['runtimeMinutes_numeric'] = df.groupby(['titleType'])['runtimeMinutes_numeric'].transform(fun_fill_median)
# If any values are still missing, fill with overall median
df['runtimeMinutes_numeric'] = df['runtimeMinutes_numeric'].fillna(df['runtimeMinutes_numeric'].median())

# %%
# Preprocessing steps from clustering script
# Extract and fill missing years based on median by title type
df['YearFill'] = df.groupby(['titleType'])['startYear'].transform(fun_fill_median)

# %%
# Create genre category and convert to numerical
df['firstGenre'] = df['genres'].str.split(',').str[0]
df['firstGenre'] = df['firstGenre'].fillna('unknown')
genres = sorted(df['firstGenre'].unique())
genre_mapping = dict(zip(genres, range(0, len(genres) + 1)))
df['genre_Val'] = df['firstGenre'].map(genre_mapping).astype(int)

# %%
# Convert title type to numerical using mapping (following professor's approach)
title_types = sorted(df['titleType'].unique())
title_type_mapping = dict(zip(title_types, range(0, len(title_types) + 1)))
df['titleType_Val'] = df['titleType'].map(title_type_mapping).astype(int)
print(f"Title type mapping: {title_type_mapping}")

# %%
# Convert country of origin to numerical
df['firstCountry'] = df['countryOfOrigin'].str.extract(r'\[\'(\w+)\'')
df['firstCountry'] = df['firstCountry'].fillna(df['firstCountry'].mode()[0])
countries = sorted(df['firstCountry'].unique())
country_mapping = dict(zip(countries, range(0, len(countries) + 1)))
df['country_Val'] = df['firstCountry'].map(country_mapping).astype(int)

# %%
# Make sure all numeric columns have proper numeric values
numeric_cols = ['runtimeMinutes', 'awardWins', 'numVotes', 'totalImages', 'totalCredits',
                'criticReviewsTotal', 'awardNominationsExcludeWins', 'numRegions', 
                'userReviewsTotal', 'ratingCount']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isnull().sum() > 0:
        # Use the grouped median approach from the professor's methodology
        df[col] = df.groupby(['titleType'])[col].transform(fun_fill_median)
        # If any values are still missing, fill with overall median
        df[col] = df[col].fillna(df[col].median() if df[col].median() is not np.nan else 0)

# %%
# Convert boolean columns to numeric
bool_cols = ['canHaveEpisodes', 'isRatable', 'isAdult']
for col in bool_cols:
    df[col] = df[col].map({'True': 1, 'False': 0})
    df[col] = df[col].fillna(0)

# %%
"""
## Compare Different Imputation Methods for Missing Values
"""

# %%
# Let's compare different imputation methods for runtimeMinutes
original_runtime = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
missing_mask = original_runtime.isnull()

# Method 1: Overall mean
runtime_overall_mean = original_runtime.fillna(original_runtime.mean())

# Method 2: Grouped mean by titleType 
runtime_grouped_mean = df.groupby(['titleType'])['runtimeMinutes_numeric'].transform(fun_fill_mean)

# Method 3: Grouped median by titleType (our chosen method)
runtime_grouped_median = df.groupby(['titleType'])['runtimeMinutes_numeric'].transform(fun_fill_median)

# %%
# Set up a grid of plots to compare the distributions
fig = plt.figure(figsize=(15, 4))
fig_dims = (1, 4)  # 1 row, 4 columns

plt.subplot2grid(fig_dims, (0, 0))
original_runtime.hist(bins=np.arange(0, 300, 10))
plt.title('Original Distribution\n(with missing)')
plt.ylim(0, 400)

plt.subplot2grid(fig_dims, (0, 1))
runtime_overall_mean.hist(bins=np.arange(0, 300, 10), color='tab:orange')
original_runtime.hist(bins=np.arange(0, 300, 10))
plt.title('Overall Mean')
plt.ylim(0, 400)

plt.subplot2grid(fig_dims, (0, 2))
runtime_grouped_mean.hist(bins=np.arange(0, 300, 10), color='tab:orange')
original_runtime.hist(bins=np.arange(0, 300, 10))
plt.title('Grouped Mean by Title Type')
plt.ylim(0, 400)

plt.subplot2grid(fig_dims, (0, 3))
runtime_grouped_median.hist(bins=np.arange(0, 300, 10), color='tab:orange')
original_runtime.hist(bins=np.arange(0, 300, 10))
plt.title('Grouped Median by Title Type')
plt.ylim(0, 400)

plt.tight_layout()
plt.show()

# %%
"""
## Correlation Analysis
"""

# %%
# Now let's check correlations between numeric features
numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
df_numeric = df[numeric_cols]

# %%
# Compute correlation matrix
correlation_matrix = df_numeric.corr(numeric_only=True)
correlation_matrix

# %%
# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# %%
"""
## Exploratory Data Visualizations
"""

# %%
"""
### Scatter Plots
Looking at bivariate data to see clusters of points, outliers, correlations
"""

# %%
"""
#### numVotes vs rating_numeric
"""

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['numVotes'], df['rating_numeric'], alpha=0.5)
plt.xlabel('Number of Votes')
plt.ylabel('Rating')
plt.title('Number of Votes vs Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
Let's try a scatter plot with a log scale for numVotes since it likely has a wide range
"""

# %%
plt.figure(figsize=(10, 6))
plt.scatter(np.log1p(df['numVotes']), df['rating_numeric'], alpha=0.5)
plt.xlabel('Log(Number of Votes + 1)')
plt.ylabel('Rating')
plt.title('Log(Number of Votes) vs Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### startYear vs rating_numeric
"""

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['startYear_numeric'], df['rating_numeric'], alpha=0.3)
plt.xlabel('Start Year')
plt.ylabel('Rating')
plt.title('Start Year vs Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### awardWins vs rating_numeric
"""

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['awardWins'], df['rating_numeric'], alpha=0.3)
plt.xlabel('Award Wins')
plt.ylabel('Rating')
plt.title('Award Wins vs Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
Let's try using Seaborn for a more sophisticated scatter plot with hue based on title type
"""

# %%
# Let's limit to the most common title types for clarity
top_title_types = df['titleType'].value_counts().nlargest(5).index.tolist()
df_top_titles = df[df['titleType'].isin(top_title_types)]

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_top_titles,
    x='numVotes',
    y='rating_numeric',
    hue='titleType',
    alpha=0.6,
    palette='viridis'
)
plt.xscale('log')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.title('Number of Votes vs Rating by Title Type')
plt.legend(title='Title Type')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
### Pairplot
"""

# %%
# Select a subset of numeric features for the pairplot to keep it manageable
numeric_features = ['rating_numeric', 'startYear_numeric', 'numVotes', 'awardWins', 'totalCredits']

# Sample a smaller subset of data to make the pairplot more efficient (if the dataset is large)
df_sample = df.sample(min(5000, len(df)))

# Create pairplot
sns.pairplot(df_sample[numeric_features], diag_kind="kde")
plt.suptitle('Pairplot of Key Numeric Features', y=1.02)
plt.show()

# %%
"""
### Histograms
Shows the frequency distribution for numerical attributes
"""

# %%
"""
#### Rating Distribution
"""

# %%
plt.figure(figsize=(10, 6))
df['rating_numeric'].hist(bins=20)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### Start Year Distribution
"""

# %%
plt.figure(figsize=(10, 6))
df['startYear_numeric'].dropna().hist(bins=30)
plt.xlabel('Start Year')
plt.ylabel('Frequency')
plt.title('Distribution of Start Years')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### Number of Votes Distribution
"""

# %%
plt.figure(figsize=(10, 6))
df['numVotes'].hist(bins=30)
plt.xlabel('Number of Votes')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Votes')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
Using log scale for numVotes
"""

# %%
plt.figure(figsize=(10, 6))
np.log1p(df['numVotes']).hist(bins=30)
plt.xlabel('Log(Number of Votes + 1)')
plt.ylabel('Frequency')
plt.title('Distribution of Log(Number of Votes)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
### Bar Charts
"""

# %%
"""
#### Count by Title Type
"""

# %%
plt.figure(figsize=(12, 6))
title_counts = df['titleType'].value_counts().nlargest(10)
title_counts.plot(kind='bar')
plt.xlabel('Title Type')
plt.ylabel('Count')
plt.title('Count of Top 10 Title Types')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.show()

# %%
"""
Using Seaborn for better visualization
"""

# %%
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='titleType', order=df['titleType'].value_counts().nlargest(10).index)
plt.xlabel('Count')
plt.ylabel('Title Type')
plt.title('Count of Top 10 Title Types')
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.show()

# %%
"""
#### Average Rating by Title Type
"""

# %%
avg_rating_by_type = df.groupby('titleType')['rating_numeric'].mean().sort_values(ascending=False).nlargest(10)

plt.figure(figsize=(12, 6))
avg_rating_by_type.plot(kind='bar')
plt.xlabel('Title Type')
plt.ylabel('Average Rating')
plt.title('Average Rating by Title Type (Top 10)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.show()

# %%
"""
### Box Plots
"""

# %%
"""
#### Rating by Title Type
"""

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='rating_numeric', y='titleType', order=df['titleType'].value_counts().nlargest(10).index)
plt.xlabel('Rating')
plt.ylabel('Title Type')
plt.title('Rating Distribution by Title Type')
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.show()

# %%
"""
#### Number of Votes by Title Type
"""

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=df, 
    x=np.log1p(df['numVotes']), 
    y='titleType', 
    order=df['titleType'].value_counts().nlargest(10).index
)
plt.xlabel('Log(Number of Votes + 1)')
plt.ylabel('Title Type')
plt.title('Log(Number of Votes) Distribution by Title Type')
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.show()

# %%
"""
### Plots Grid
"""

# %%
# Set up a grid of plots
fig = plt.figure(figsize=(15, 10))
fig_dims = (2, 3)  # 2 rows, 3 columns

# Plot 1: Rating distribution
plt.subplot2grid(fig_dims, (0, 0))
df['rating_numeric'].hist()
plt.title('Rating Distribution')

# Plot 2: Title type counts (top 5)
plt.subplot2grid(fig_dims, (0, 1))
df['titleType'].value_counts().nlargest(5).plot(kind='bar')
plt.title('Top 5 Title Types')
plt.xticks(rotation=45)

# Plot 3: Start year distribution
plt.subplot2grid(fig_dims, (0, 2))
df['startYear_numeric'].dropna().hist()
plt.title('Start Year Distribution')

# Plot 4: Number of votes distribution (log scale)
plt.subplot2grid(fig_dims, (1, 0))
np.log1p(df['numVotes']).hist()
plt.title('Log(Number of Votes) Distribution')

# Plot 5: Award wins distribution
plt.subplot2grid(fig_dims, (1, 1))
df['awardWins'].hist()
plt.title('Award Wins Distribution')

# Plot 6: Total credits distribution
plt.subplot2grid(fig_dims, (1, 2))
df['totalCredits'].hist()
plt.title('Total Credits Distribution')

plt.tight_layout()
plt.show()

# %%
"""
### Genres Analysis
"""

# %%
# Check if there's a genres column in the dataset
if 'genres' in df.columns:
    # Extract genres and create a new DataFrame for analysis
    genres_list = []
    
    # Extract all genres and count their occurrences
    for genres in df['genres'].dropna():
        if isinstance(genres, str):
            # If genres is stored as a string representation of a list, parse it
            if genres.startswith('[') and genres.endswith(']'):
                genres = genres.strip('[]').replace("'", "").split(',')
            # If genres is stored as comma-separated values
            else:
                genres = genres.split(',')
                
            for genre in genres:
                genre = genre.strip()
                if genre:
                    genres_list.append(genre)
    
    # Count genres
    genres_count = pd.Series(genres_list).value_counts()
    
    # Plot top 20 genres
    plt.figure(figsize=(12, 8))
    genres_count.nlargest(20).plot(kind='bar')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Top 20 Genres')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Calculate average rating by genre
    if 'rating_numeric' in df.columns:
        genre_avg_ratings = {}
        genre_counts = {}
        
        for index, row in df.iterrows():
            if pd.notna(row['genres']) and pd.notna(row['rating_numeric']):
                genres_str = row['genres']
                
                # Parse genres string
                if isinstance(genres_str, str):
                    if genres_str.startswith('[') and genres_str.endswith(']'):
                        genres = genres_str.strip('[]').replace("'", "").split(',')
                    else:
                        genres = genres_str.split(',')
                        
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            if genre not in genre_avg_ratings:
                                genre_avg_ratings[genre] = 0
                                genre_counts[genre] = 0
                            
                            genre_avg_ratings[genre] += row['rating_numeric']
                            genre_counts[genre] += 1
        
        # Calculate average
        for genre in genre_avg_ratings:
            if genre_counts[genre] > 0:
                genre_avg_ratings[genre] /= genre_counts[genre]
        
        # Convert to Series for easier plotting
        genre_avg_ratings = pd.Series(genre_avg_ratings)
        
        # Plot average rating for top 20 genres by count
        plt.figure(figsize=(12, 8))
        genre_avg_ratings[genres_count.nlargest(20).index].sort_values(ascending=False).plot(kind='bar')
        plt.xlabel('Genre')
        plt.ylabel('Average Rating')
        plt.title('Average Rating by Genre (Top 20 by Count)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()

# %%
"""
### Time Series Analysis
"""

# %%
# Analyze how ratings changed over time
if 'startYear_numeric' in df.columns and 'rating_numeric' in df.columns:
    # Group by year and calculate mean rating
    yearly_ratings = df.groupby('startYear_numeric')['rating_numeric'].mean()
    
    # Plot the trend
    plt.figure(figsize=(15, 6))
    yearly_ratings.plot()
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.title('Average Rating by Year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot the count of titles by year
    yearly_counts = df.groupby('startYear_numeric').size()
    
    plt.figure(figsize=(15, 6))
    yearly_counts.plot()
    plt.xlabel('Year')
    plt.ylabel('Number of Titles')
    plt.title('Number of Titles by Year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# %%
"""
### Cross Tab Analysis
"""

# %%
# Let's create a categorical variable for rating ranges
if 'rating_numeric' in df.columns:
    df['rating_class'] = pd.qcut(
        df['rating_numeric'], 
        q=3, 
        labels=[0, 1, 2]
    )
    
    # Create a categorical variable for decades
    if 'startYear_numeric' in df.columns:
        df['decade'] = (df['startYear_numeric'] // 10) * 10
        
        # Cross-tabulation of decades and rating classes
        decade_rating_crosstab = pd.crosstab(df['decade'], df['rating_class'])
        
        # Normalize to get percentages
        decade_rating_pct = decade_rating_crosstab.div(decade_rating_crosstab.sum(axis=1), axis=0)
        
        # Plot the stacked bar chart
        plt.figure(figsize=(15, 8))
        decade_rating_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.xlabel('Decade')
        plt.ylabel('Percentage')
        plt.title('Rating Distribution by Decade')
        plt.legend(title='Rating Class')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()

# %%
"""
## Part 2: Clustering Analysis
"""

# %%
"""
### Preparing Data for Clustering
"""

# %%
# Drop non-essential columns and prepare data for clustering
df_train = df.drop(['originalTitle', 'rating', 'titleType', 'countryOfOrigin', 'genres', 'firstCountry', 'firstGenre', 'endYear'], axis=1)
df_train = df_train.drop(['worstRating', 'bestRating', 'startYear', 'isRatable', 'canHaveEpisodes', 'totalVideos'], axis=1)

# Check for any remaining non-numeric columns
for col in df_train.columns:
    if df_train[col].dtype == 'object':
        print(f"Column {col} is still object type and will be dropped")
        df_train = df_train.drop(col, axis=1)

# Convert any remaining NaN values to 0
df_train = df_train.fillna(0)

print("Data types after preprocessing:")
print(df_train.dtypes)

# Check for any remaining NaN values
print("\nRemaining NaN values:")
print(df_train.isnull().sum())

# %%
df_train.head()

# %%
# Scale the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(df_train)
print(f"Scaling complete. Train data shape: {train_data.shape}")

# %%
"""
### K-Means Clustering
"""

# %%
# Calculate optimal number of clusters using elbow method and silhouette score
sse_list = []
sil_list = []

print("Calculating optimal number of clusters...")
for k in range(2, 51):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100, random_state=42)
    kmeans.fit(train_data)
    sse_list.append(kmeans.inertia_)
    sil_list.append(silhouette_score(train_data, kmeans.labels_))
    if k % 10 == 0:
        print(f"Completed k={k}")

# %%
# Plot SSE and silhouette scores
fig, axs = plt.subplots(2) # 1 row, 2 columns

sns.lineplot(x=range(len(sse_list)), y=sse_list, marker='o', ax=axs[0])
axs[0].set(xlabel='k', ylabel='SSE')
sns.lineplot(x=range(len(sil_list)), y=sil_list, marker='o', ax=axs[1])
axs[1].set(xlabel='k', ylabel='Silhouette')

plt.tight_layout() # Adjust the padding between and around subplots
plt.show()

# %%
# Find optimal k based on silhouette score
best_k_idx = np.argmax(sil_list)
optimal_k = best_k_idx + 2  # +2 because we start from k=2
print(f"Optimal k based on silhouette score: {optimal_k}")

kmeans = KMeans(init='k-means++', n_clusters=optimal_k, n_init=10, max_iter=100, random_state=42)
kmeans.fit(train_data)

# %%
# Print clustering metrics
print('labels', np.unique(kmeans.labels_, return_counts=True))
print('sse', kmeans.inertia_)
print('silhouette', silhouette_score(train_data, kmeans.labels_))

# %%
# Plot cluster centers
plt.figure(figsize=(10, 3))

for i in range(len(kmeans.cluster_centers_)):
    plt.plot(range(df_train.shape[1]), kmeans.cluster_centers_[i], label='Cluster %s' % i, linewidth=3)
plt.xticks(range(df_train.shape[1]), list(df_train.columns), rotation=45)
plt.legend(bbox_to_anchor=(1,1))
plt.grid(axis='y')
plt.show()

# %%
# Add cluster labels to the dataframe for analysis
df_clusters = df_train.copy()
df_clusters['Labels'] = kmeans.labels_

# %%
# Visualize clusters with scatter plot
sns.scatterplot(data=df_clusters, 
                x="YearFill",
                y="numVotes", 
                hue=kmeans.labels_, 
                style=kmeans.labels_, 
                palette="bright")
plt.yscale('log')  # Using log scale for better visualization
plt.show()

# %%
# Cross tabulation of title types and clusters
titletype_xt = pd.crosstab(df['titleType'], df_clusters['Labels'])
titletype_xt

# %%
# Cross tabulation of genres and clusters
genre_xt = pd.crosstab(df_clusters['Labels'], df['firstGenre'])
genre_xt

# %%
# Visualize genre distribution in clusters
plt.figure(figsize=(12, 8))  # Increase figure size
genre_xt_pct = genre_xt.div(genre_xt.sum(1).astype(float), axis=0)
genre_xt_pct.plot(kind='bar', stacked=True, title='Cluster Label Rate by Genres')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.legend(title='firstGenre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()

# %%
# Analyze one of the clusters
df_clusters[df_clusters['Labels']==1].describe()

# %%
# Visualize title type distribution in clusters
plt.figure(figsize=(12, 8))  # Increase figure size
titletype_xt_pct = titletype_xt.div(titletype_xt.sum(1).astype(float), axis=0)
titletype_xt_pct.plot(kind='bar', stacked=True, title='Cluster Label Rate by Title Types')
plt.xlabel('Title Type')
plt.ylabel('Proportion')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()

# %%
"""
### DBSCAN Clustering
"""

# %%
# Set environment variable for parallel processing as in the original clustering file
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# %%
"""
### Prepare Better Features for DBSCAN
"""

# %%
# Select more discriminative features for DBSCAN
print("Selecting more discriminative features for DBSCAN...")
features_dbscan = ['runtimeMinutes_numeric', 'startYear_numeric', 'numVotes', 'totalImages', 
                   'awardWins', 'criticReviewsTotal', 'totalCredits',
                   'genre_Val', 'titleType_Val']  # Include categorical values converted to numeric

# Select samples to work with (using more samples for better clustering)
X_dbscan = df[features_dbscan].copy()

# Scale the data for DBSCAN (standardizing is better than min-max for DBSCAN)
scaler = StandardScaler()  # Changed from MinMaxScaler to StandardScaler
X_dbscan_scaled = scaler.fit_transform(X_dbscan)

print(f"Prepared {X_dbscan.shape[0]} samples with {X_dbscan.shape[1]} features for DBSCAN")

# %%
"""
### Feature Analysis for DBSCAN
"""

# %%
# Check feature variance to identify which features might be most useful
feature_variance = pd.DataFrame({
    'Feature': features_dbscan,
    'Variance': np.var(X_dbscan_scaled, axis=0)
})
feature_variance = feature_variance.sort_values('Variance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Variance', y='Feature', data=feature_variance)
plt.title('Feature Variance (Higher is Better for Clustering)')
plt.tight_layout()
plt.show()

print("Feature variance analysis:")
print(feature_variance)

# %%
"""
### Parameter Selection for DBSCAN

Following the professor's methodology, we'll explore different parameters for DBSCAN
"""

# %%
# Calculate nearest neighbors distances to help determine eps
from sklearn.neighbors import NearestNeighbors

# Use a sample if the dataset is too large
if len(X_dbscan_scaled) > 5000:
    sample_indices = np.random.choice(len(X_dbscan_scaled), 5000, replace=False)
    X_sample = X_dbscan_scaled[sample_indices]
else:
    X_sample = X_dbscan_scaled

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X_sample)
distances, indices = nbrs.kneighbors(X_sample)

# Sort distances to find the "elbow"
distances = np.sort(distances[:, 1])

# %%
# Plot the distances to help find a good eps value
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('Distance to 2nd nearest neighbor')
plt.title('Distance to 2nd Nearest Neighbor (sorted)')
plt.grid(True)
plt.show()

# %%
# Plot a zoom-in of the distances for better visibility
plt.figure(figsize=(10, 6))
plt.plot(distances[:int(len(distances) * 0.2)])  # Plot only the first 20% of points
plt.xlabel('Points')
plt.ylabel('Distance to 2nd nearest neighbor')
plt.title('Distance to 2nd Nearest Neighbor (first 20%)')
plt.grid(True)
plt.show()

# %%
"""
### Improved Method for Determining Optimal Epsilon
"""

# %%
# Calculate k-distance graph to help determine eps (using a sample if needed)
from sklearn.neighbors import NearestNeighbors

# Define a better value for k - rule of thumb is 2*dimensions
k = min(2 * X_dbscan_scaled.shape[1], X_dbscan_scaled.shape[0] // 10)
print(f"Using k={k} for nearest neighbors distance calculation")

# For large datasets, use a sample
if X_dbscan_scaled.shape[0] > 5000:
    print(f"Using a sample of 5000 points from {X_dbscan_scaled.shape[0]} total points")
    sample_indices = np.random.choice(X_dbscan_scaled.shape[0], 5000, replace=False)
    X_sample = X_dbscan_scaled[sample_indices]
else:
    print(f"Using all {X_dbscan_scaled.shape[0]} points for distance calculation")
    X_sample = X_dbscan_scaled

# Calculate distances to k-nearest neighbors
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X_sample)
distances, indices = nbrs.kneighbors(X_sample)

# Sort the distances to the kth nearest neighbor (last column)
kth_distances = distances[:, -1]
kth_distances.sort()

# Plot the k-distance graph
plt.figure(figsize=(12, 8))
plt.plot(kth_distances)
plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3, label='eps=0.5')
plt.axhline(y=0.3, color='g', linestyle='-', alpha=0.3, label='eps=0.3')
plt.axhline(y=0.2, color='b', linestyle='-', alpha=0.3, label='eps=0.2')
plt.axhline(y=0.1, color='m', linestyle='-', alpha=0.3, label='eps=0.1')
plt.xlabel(f'Points (sorted by distance to {k}th nearest neighbor)')
plt.ylabel(f'Distance to {k}th nearest neighbor')
plt.title(f'K-Distance Graph (k={k})')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Find the "elbow point" in the k-distance graph
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

try:
    # Calculate first derivative to find the rate of change
    x = np.arange(len(kth_distances))
    # Smoothen the data with a moving average to reduce noise
    window_size = max(10, len(kth_distances) // 100)
    smoothed_distances = np.convolve(kth_distances, np.ones(window_size)/window_size, mode='valid')
    x_smooth = x[window_size-1:]
    
    # Find potential elbow points (local maxima of first derivative)
    diffs = np.diff(smoothed_distances)
    # Find local maxima in the derivative (points of maximum curvature)
    local_max_indices = argrelextrema(diffs, np.greater_equal, order=window_size)[0]
    
    if len(local_max_indices) > 0:
        # Choose a point with significant rate change
        threshold = np.percentile(diffs, 95)  # 95th percentile of derivative values
        significant_changes = local_max_indices[diffs[local_max_indices] > threshold]
        
        if len(significant_changes) > 0:
            elbow_idx = significant_changes[0]  # Take the first significant change point
            suggested_eps = smoothed_distances[elbow_idx]
            
            plt.figure(figsize=(12, 8))
            plt.plot(kth_distances, label='K-distance')
            plt.plot(x_smooth[elbow_idx], suggested_eps, 'ro', markersize=10, label=f'Suggested eps={suggested_eps:.3f}')
            plt.axhline(y=suggested_eps, color='r', linestyle='--', alpha=0.5)
            plt.xlabel(f'Points (sorted by distance to {k}th nearest neighbor)')
            plt.ylabel(f'Distance to {k}th nearest neighbor')
            plt.title(f'K-Distance Graph with Suggested Epsilon (k={k})')
            plt.grid(True)
            plt.legend()
            plt.show()
            
            print(f"Suggested epsilon value: {suggested_eps:.3f}")
        else:
            print("No significant elbow point found in the k-distance graph")
    else:
        print("No local maxima found in the k-distance graph")
except Exception as e:
    print(f"Error finding elbow point: {e}")
    print("Unable to automatically determine optimal epsilon value")

# %%
"""
### Testing Different eps Values for DBSCAN
"""

# %%
# Define a wider range of eps values to try
eps_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
min_samples_values = [5, 10, 15, 20]

# Store results for different parameter combinations
dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        # Apply DBSCAN with the current parameters
        dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))  # Ensure min_samples is an integer
        labels = dbscan.fit_predict(X_dbscan_scaled)
        
        # Count the number of clusters (excluding noise points which are labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate silhouette score if there are multiple clusters
        sil_score = np.nan
        if n_clusters > 1 and len(set(labels)) > 1:
            # Exclude noise points for silhouette score calculation
            mask = labels != -1
            if sum(mask) > 1 and len(set(labels[mask])) > 1:
                try:
                    sil_score = silhouette_score(X_dbscan_scaled[mask], labels[mask])
                except Exception as e:
                    print(f"  Error calculating silhouette score: {e}")
        
        # Add results to the list
        dbscan_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': n_noise / len(labels) * 100,
            'silhouette_score': sil_score
        })
        
# %%
# Convert results to DataFrame for easier analysis
dbscan_results_df = pd.DataFrame(dbscan_results)
dbscan_results_df.sort_values(by=['eps', 'min_samples'], inplace=True)
dbscan_results_df

# %%
# Visualize how the number of clusters changes with eps and min_samples
plt.figure(figsize=(12, 8))
has_valid_data = False  # Track whether we have any valid data to plot

for min_samples in min_samples_values:
    subset = dbscan_results_df[dbscan_results_df['min_samples'] == min_samples]
    
    # Check if subset has any rows and if n_clusters has variation
    if not subset.empty and subset['n_clusters'].nunique() > 1:
        plt.plot(subset['eps'], subset['n_clusters'], marker='o', linewidth=2, 
                 markersize=8, label=f'min_samples={min_samples}')
        has_valid_data = True
    elif not subset.empty:
        # Still plot flat lines but with a note
        plt.plot(subset['eps'], subset['n_clusters'], marker='o', linewidth=2, 
                 markersize=8, label=f'min_samples={min_samples} (constant)')
        has_valid_data = True

# Add labels and title with appropriate context
plt.xlabel('Epsilon (eps)', fontsize=12)
plt.ylabel('Number of Clusters', fontsize=12)

if has_valid_data:
    plt.title('Number of Clusters vs. eps for Different min_samples Values', fontsize=14)
    plt.legend()
else:
    plt.title('Number of Clusters vs. eps (No variation in cluster numbers found)', fontsize=14)
    plt.text(0.5, 0.5, 'Try different parameters for more clusters', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)

plt.grid(True)
plt.show()

# %%
# Visualize how the noise percentage changes with eps and min_samples
plt.figure(figsize=(12, 8))
has_valid_data = False  # Track whether we have any valid data to plot

for min_samples in min_samples_values:
    subset = dbscan_results_df[dbscan_results_df['min_samples'] == min_samples]
    
    if not subset.empty:
        plt.plot(subset['eps'], subset['noise_percentage'], marker='o', linewidth=2, 
                 markersize=8, label=f'min_samples={min_samples}')
        has_valid_data = True

if has_valid_data:
    plt.xlabel('Epsilon (eps)', fontsize=12)
    plt.ylabel('Noise Percentage (%)', fontsize=12)
    plt.title('Noise Percentage vs. eps for Different min_samples Values', fontsize=14)
    plt.legend()
    plt.grid(True)
else:
    plt.title('Noise Percentage vs. eps (No data available)', fontsize=14)
    plt.text(0.5, 0.5, 'No valid noise percentage data found', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)

plt.show()

# %%
# Visualize silhouette scores for different parameter combinations
plt.figure(figsize=(12, 8))
has_valid_scores = False  # Track whether we have any valid silhouette scores

for min_samples in min_samples_values:
    subset = dbscan_results_df[dbscan_results_df['min_samples'] == min_samples]
    valid_scores = subset[~subset['silhouette_score'].isna()]
    
    if not valid_scores.empty:
        plt.plot(valid_scores['eps'], valid_scores['silhouette_score'], marker='o', 
                 linewidth=2, markersize=8, label=f'min_samples={min_samples}')
        has_valid_scores = True

if has_valid_scores:
    plt.xlabel('Epsilon (eps)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs. eps for Different min_samples Values', fontsize=14)
    plt.legend()
    plt.grid(True)
else:
    plt.title('Silhouette Score vs. eps (No valid scores available)', fontsize=14)
    plt.text(0.5, 0.5, 'No valid silhouette scores found\nTry different parameters or larger dataset', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)

plt.show()

# %%
"""
### Selecting the Best DBSCAN Parameters
"""

# %%
# Find the parameters with the best silhouette score
try:
    # Check if there are any valid silhouette scores
    valid_scores = dbscan_results_df[~dbscan_results_df['silhouette_score'].isna()]
    
    if valid_scores.empty:
        print("No valid silhouette scores found. Using default parameters.")
        best_eps = 0.5
        best_min_samples = 5
    else:
        # Find the row with the maximum silhouette score
        best_params = valid_scores.loc[valid_scores['silhouette_score'].idxmax()]
        best_eps = best_params['eps']
        best_min_samples = int(best_params['min_samples'])  # Explicitly convert to int
        
        print(f"Best Parameters:\neps: {best_eps}\nmin_samples: {best_min_samples}")
        print(f"Number of Clusters: {best_params['n_clusters']}")
        print(f"Noise Percentage: {best_params['noise_percentage']:.2f}%")
        print(f"Silhouette Score: {best_params['silhouette_score']:.4f}")
except Exception as e:
    print(f"Error in parameter selection: {e}")
    print("Using default parameters.")
    best_eps = 0.5
    best_min_samples = 5

# %%
# Apply DBSCAN with the selected parameters
dbscan = DBSCAN(eps=best_eps, min_samples=int(best_min_samples))  # Ensure min_samples is an integer
dbscan_labels = dbscan.fit_predict(X_dbscan_scaled)

# Count the number of clusters and noise points
unique_labels = set(dbscan_labels)
n_clusters = len(unique_labels) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Number of estimated clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.2f}%)")
print(f"Total number of points: {len(dbscan_labels)}")

# %%
# Add the cluster labels to the original dataframe
df['dbscan_cluster'] = dbscan_labels

# %%
"""
### Visualizing DBSCAN Clusters
"""

# %%
"""
### Visualizing DBSCAN Clusters in 2D
"""

# %%
# Visualize clusters in 2D (using PCA for dimensionality reduction)
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dbscan_scaled)

# Create a more visually appealing plot
plt.figure(figsize=(12, 10))

# Define a colormap with a reserved color for noise points
unique_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
colors = plt.cm.viridis(np.linspace(0, 1, unique_clusters))

# Plot each cluster with a different color
for i, label in enumerate(sorted(list(set(dbscan_labels)))):
    if label == -1:
        # Plot noise points as black x's
        mask = dbscan_labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=40, 
                    c='black', marker='x', alpha=0.5,
                    label='Noise')
    else:
        # Plot cluster points
        mask = dbscan_labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=80, 
                    alpha=0.8, marker='o',
                    label=f'Cluster {label}')

# Add title and labels with information about the parameters
plt.title(f'DBSCAN Clustering (eps={best_eps}, min_samples={best_min_samples})\n'
          f'{n_clusters} clusters, {n_noise} noise points ({n_noise/len(dbscan_labels)*100:.1f}%)',
          fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Add a legend with proper sizing
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Add gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Display explained variance by the principal components
explained_variance = pca.explained_variance_ratio_
print(f"Variance explained by PC1: {explained_variance[0]:.2%}")
print(f"Variance explained by PC2: {explained_variance[1]:.2%}")
print(f"Total variance explained: {sum(explained_variance):.2%}")

# %%
"""
### Cluster Analysis for DBSCAN
"""

# %%
# Analyze the distribution of title types within each cluster
dbscan_title_cross = pd.crosstab(df['titleType'], df['dbscan_cluster'])
dbscan_title_cross

# %%
# Normalize the cross-tabulation to see percentages
dbscan_title_cross_norm = pd.crosstab(df['titleType'], df['dbscan_cluster'], normalize='columns') * 100
dbscan_title_cross_norm

# %%
# Visualize the distribution of title types within each cluster
plt.figure(figsize=(14, 8))
dbscan_title_cross_norm.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('Distribution of Title Types in DBSCAN Clusters')
plt.xlabel('Title Type')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Analyze the distribution of genres within each cluster
dbscan_genre_cross = pd.crosstab(df['firstGenre'], df['dbscan_cluster'])
dbscan_genre_cross

# %%
# Normalize the genre cross-tabulation
dbscan_genre_cross_norm = pd.crosstab(df['firstGenre'], df['dbscan_cluster'], normalize='columns') * 100
dbscan_genre_cross_norm

# %%
# Visualize the distribution of genres within each cluster
plt.figure(figsize=(14, 8))
dbscan_genre_cross_norm.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('Distribution of Genres in DBSCAN Clusters')
plt.xlabel('Genre')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Analyze numeric features for each cluster
dbscan_cluster_means = df.groupby('dbscan_cluster')[features_dbscan].mean()
dbscan_cluster_means

# %%
# Visualize the cluster profiles
plt.figure(figsize=(14, 8))
dbscan_cluster_means.T.plot(kind='bar', colormap='tab10')
plt.title('DBSCAN Cluster Profiles (Mean Values of Features)')
plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Calculate min, max, and std for each feature across clusters
dbscan_cluster_min = df.groupby('dbscan_cluster')[features_dbscan].min()
dbscan_cluster_max = df.groupby('dbscan_cluster')[features_dbscan].max()
dbscan_cluster_std = df.groupby('dbscan_cluster')[features_dbscan].std()

# %%
# Print statistical summary for each cluster
print("Statistical Summary for each DBSCAN Cluster:")
print("\nMean Values:")
print(dbscan_cluster_means)
print("\nStandard Deviation:")
print(dbscan_cluster_std)
print("\nMinimum Values:")
print(dbscan_cluster_min)
print("\nMaximum Values:")
print(dbscan_cluster_max)

# %%
"""
## Hierarchical Clustering
"""

# %%
# Helper functions for hierarchical clustering visualization
def get_linkage_matrix(model):
    # Create linkage matrix 
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

def plot_dendrogram(model, **kwargs):
    linkage_matrix = get_linkage_matrix(model)
    dendrogram(linkage_matrix, **kwargs)

# %%
# Prepare data for hierarchical clustering
print("Preparing data for hierarchical clustering...")
# Select features for hierarchical clustering
features_hierarchical = ['runtimeMinutes_numeric', 'startYear_numeric', 'awardWins', 'numVotes']

# Take a sample for hierarchical clustering to avoid memory issues
np.random.seed(42)
sample_size = min(1000, len(df))
sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
X_hierarchical = df.iloc[sample_indices][features_hierarchical].values

# Scale the data
scaler_hierarchical = StandardScaler()
X_hierarchical_scaled = scaler_hierarchical.fit_transform(X_hierarchical)

# setting distance_threshold=0 ensures we compute the full tree
print("Applying hierarchical clustering on a sample...")
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric='euclidean', linkage='complete')
model = model.fit(X_hierarchical_scaled)
print("Hierarchical clustering complete.")

# %%
# Plotting the dendrogram
try:
    plt.figure(figsize=(14, 8))
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="lastp", color_threshold=1.2, p=30)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error plotting dendrogram: {str(e)}")
    # Fallback to simpler dendrogram if needed
    plt.figure(figsize=(14, 8))
    plt.title("Hierarchical Clustering Dendrogram (Simple)")
    Z = get_linkage_matrix(model)
    dendrogram(Z, truncate_mode="lastp", p=30)
    plt.xlabel("Number of points in node")
    plt.grid(True)
    plt.show()

# %%
# Get the labels according to a specific threshold value cut
Z = get_linkage_matrix(model)
threshold = 1.2  # Adjust this threshold based on the dendrogram
print(f"Cutting dendrogram at threshold: {threshold}")
labels = fcluster(Z, t=threshold, criterion='distance')

# Calculate silhouette score for the hierarchical clusters
try:
    sil_score = silhouette_score(X_hierarchical_scaled, labels)
    print(f'Silhouette score: {sil_score:.4f}')
except Exception as e:
    print(f"Error calculating silhouette score: {str(e)}")

# %%
"""
### Finding Optimal Number of Clusters for Hierarchical Clustering
"""

# %%
# Determine optimal number of clusters using silhouette scores
max_clusters = 10
silhouette_scores = []
for n_clusters in range(2, max_clusters + 1):
    hier = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='complete')
    hier_labels = hier.fit_predict(X_hierarchical_scaled)
    
    try:
        sil_score = silhouette_score(X_hierarchical_scaled, hier_labels)
        silhouette_scores.append(sil_score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {sil_score:.4f}")
    except Exception as e:
        print(f"Error with {n_clusters} clusters: {str(e)}")
        silhouette_scores.append(np.nan)

# %%
# Plot silhouette scores for different numbers of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.grid(True)
plt.show()

# %%
# Choose the optimal number of clusters based on silhouette score
try:
    optimal_clusters = np.nanargmax(silhouette_scores) + 2  # +2 because we started from 2 clusters
    print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
except Exception as e:
    print(f"Error finding optimal clusters: {str(e)}")
    optimal_clusters = 3  # fallback to a default value

# %%
# Apply hierarchical clustering with the optimal number of clusters
hier_optimal = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='complete')
hier_optimal.fit(X_hierarchical_scaled)

# Add the cluster labels to the sample data
df_sample = df.iloc[sample_indices].copy()
df_sample['hierarchical_cluster'] = hier_optimal.labels_

# %%
# Visualize the clusters using PCA
pca_hier = PCA(n_components=2)
X_hier_pca = pca_hier.fit_transform(X_hierarchical_scaled)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_hier_pca[:, 0], X_hier_pca[:, 1], 
                     c=hier_optimal.labels_, 
                     cmap='viridis', 
                     s=50, 
                     alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Hierarchical Clustering with {optimal_clusters} Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# %%
"""
### Hierarchical Clustering with Connectivity Constraints
"""

# %%
# Compute connectivity matrix for the sample data
print("Computing connectivity constraints for sample data...")
sample_connectivity = kneighbors_graph(
    X_hierarchical_scaled, 
    n_neighbors=20,            # Number of neighbors
    include_self=False,
    n_jobs=-1                  # Use all available CPU cores
)
print(f"Connectivity matrix shape: {sample_connectivity.shape}")
print(f"Sample data shape: {X_hierarchical_scaled.shape}")

# %%
# Apply hierarchical clustering with connectivity constraints
try:
    print("Fitting hierarchical clustering with connectivity constraints...")
    model_connected = AgglomerativeClustering(
        n_clusters=optimal_clusters,
        metric='euclidean', 
        linkage='ward', 
        connectivity=sample_connectivity  # Use the connectivity matrix
    )
                             
    model_connected = model_connected.fit(X_hierarchical_scaled)
    print("Fitting successful!")

    # Calculate silhouette score for the connected model
    try:
        sil_score_connected = silhouette_score(X_hierarchical_scaled, model_connected.labels_)
        print(f'Silhouette score with connectivity: {sil_score_connected:.4f}')
    except Exception as e:
        print(f"Error calculating silhouette score: {str(e)}")
    
    # Visualize the connected clusters using PCA
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_hier_pca[:, 0], X_hier_pca[:, 1], 
                         c=model_connected.labels_, 
                         cmap='viridis', 
                         s=50, 
                         alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Hierarchical Clustering with Connectivity Constraints ({optimal_clusters} Clusters)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"Error with connectivity constraint: {str(e)}")

# %%
"""
### Comparing Different Linkage Methods with Connectivity Constraints
"""

# %%
# Try different linkage methods with connectivity constraints
linkage_methods = ['ward', 'average', 'complete']
linkage_results = []

for linkage_method in linkage_methods:
    print(f"Testing {linkage_method} linkage method...")
    
    try:
        # Apply clustering with the current linkage method
        hierc = AgglomerativeClustering(
            n_clusters=optimal_clusters, 
            linkage=linkage_method, 
            connectivity=sample_connectivity
        )
        hierc.fit(X_hierarchical_scaled)
        
        # Calculate label distribution
        unique_labels, counts = np.unique(hierc.labels_, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        
        # Calculate silhouette score
        sil_score_method = silhouette_score(X_hierarchical_scaled, hierc.labels_)
        
        # Add results to the list
        linkage_results.append({
            'linkage_method': linkage_method,
            'label_distribution': label_distribution,
            'silhouette_score': sil_score_method
        })
        
        print(f"  Labels: {label_distribution}")
        print(f"  Silhouette score: {sil_score_method:.4f}")
        
        # Visualize clusters using PCA
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_hier_pca[:, 0], X_hier_pca[:, 1], 
                             c=hierc.labels_, 
                             cmap='viridis', 
                             s=50, 
                             alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Hierarchical Clustering with {linkage_method.capitalize()} Linkage ({optimal_clusters} Clusters)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"  Error with {linkage_method} linkage: {str(e)}")

# %%
# Compare silhouette scores for different linkage methods
if linkage_results:
    linkage_df = pd.DataFrame(linkage_results)
    plt.figure(figsize=(10, 6))
    plt.bar(linkage_df['linkage_method'], linkage_df['silhouette_score'])
    plt.xlabel('Linkage Method')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Linkage Methods')
    plt.ylim(0, max(linkage_df['silhouette_score']) * 1.2)
    plt.grid(True, axis='y')
    plt.show()

# %%
"""
## Categorical & Mixed Distance Clustering
"""

# %%
# Prepare data for categorical and mixed distance analysis
cols2drop = ['originalTitle', 'titleType_Val', 'country_Val', 'genre_Val', 'startYear', 'endYear', 'rating']
df_xm = df.drop(cols2drop, axis=1, errors='ignore')
df_xm['titleType'] = df_xm['titleType'].astype(str)

# %%
# Create dummy variables for categorical columns
categorical_cols = ['titleType', 'firstGenre', 'firstCountry']
df_xm2 = pd.get_dummies(df_xm[categorical_cols], prefix_sep='=')
df_xm2.head()

# %%
# Extract and use a subset of categorical columns for distance calculation
# Limit to first 20 columns to avoid memory issues
cat_cols = [col for col in df_xm2.columns if col.startswith('titleType=') or col.startswith('firstGenre=')][:20]
print(f"Using {len(cat_cols)} categorical columns: {cat_cols}")

# Use a sample of data for demonstration
sample_size_cat = min(1000, len(df_xm2))
sample_indices_cat = np.random.choice(len(df_xm2), size=sample_size_cat, replace=False)
X_cat = df_xm2.loc[sample_indices_cat, cat_cols].values

# %%
# Display first few rows of the data
print("Sample of categorical data (first 5 rows, first 10 columns):")
print(X_cat[:5, :10])

# %%
# Calculate Jaccard distance for categorical data
print("Calculating Jaccard distances...")
D = pdist(X_cat, 'jaccard')
D_square = squareform(D)

# %%
# Show a sample of the distance matrix
print("Sample of Jaccard distance matrix (first 5x5):")
print(D_square[:5, :5])

# %%
"""
### Mixed Custom Distance Function
"""

# %%
# Define a mixed distance function that combines Euclidean for numeric and Jaccard for categorical
def mixed(a, b, numeric_index=4):
    """
    Calculate mixed distance between two data points.
    
    Parameters:
    - a, b: Data points to compare
    - numeric_index: Number of numeric features at the beginning of the vectors
    
    Returns:
    - Combined distance
    """
    # Convert first part of arrays to float to avoid boolean subtraction error
    a_numeric = a[:numeric_index].astype(float)
    b_numeric = b[:numeric_index].astype(float)
    
    # Calculate distance for numeric part using seuclidean
    d_con = seuclidean(a_numeric, b_numeric, V=np.ones(numeric_index))
    w_con = numeric_index/len(a)
    
    # Calculate distance for categorical part using jaccard
    if np.any(a[numeric_index:]) or np.any(b[numeric_index:]):
        # Only calculate jaccard if either vector has non-zero values
        d_cat = jaccard(a[numeric_index:].astype(bool), b[numeric_index:].astype(bool))
    else:
        # If both vectors are all zeros, distance is 0
        d_cat = 0.0
        
    w_cat = (len(a)-numeric_index)/len(a)
    
    # Combine distances
    d = w_con * d_con + w_cat * d_cat
    return d

# %%
# Create a mixed dataset with both numeric and categorical data
# Select 4 numeric features
numeric_features = ['numVotes', 'awardWins', 'totalImages', 'totalCredits']
numeric_sample = df.loc[sample_indices_cat, numeric_features].values

# Combine with categorical data
X_mixed = np.hstack((numeric_sample, X_cat))

# Scale the numeric part
scaler_mixed = StandardScaler()
X_mixed[:, :4] = scaler_mixed.fit_transform(X_mixed[:, :4])

# %%
# Demonstrate mixed distance calculation
print("Example of mixed distance calculation:")
print(f"Distance between point 0 and point 5: {mixed(X_mixed[0], X_mixed[5])}")
print(f"Distance between point 1 and point 9: {mixed(X_mixed[1], X_mixed[9])}")

# %%
# Calculate distance matrix using mixed distance
try:
    print(f"Calculating mixed distances for {len(X_mixed)} points...")
    D_mixed = pdist(X_mixed, mixed)
    D_mixed_square = squareform(D_mixed)
    
    print("Mixed distance matrix shape:", D_mixed_square.shape)
    print("Sample of mixed distance matrix:")
    print(D_mixed_square[:5, :5])  # Show a small sample
except Exception as e:
    print(f"Error calculating mixed distances: {str(e)}")

# %%
"""
### K-Modes Clustering for Categorical Data
"""

# %%
try:
    from kmodes.kmodes import KModes
    
    print("Preparing data for K-Modes...")
    # Use categorical columns only
    X_kmodes = df.loc[sample_indices_cat, categorical_cols].astype(str).values
    
    print("Sample data for K-Modes:")
    print(X_kmodes[:5])
    
    print("Applying K-Modes clustering...")
    km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(X_kmodes)
    
    # Print cluster centers
    print("\nCluster centers:")
    for i, center in enumerate(km.cluster_centroids_):
        print(f"Cluster {i} center: {center}")
    
    # Print cluster labels distribution
    unique_labels, counts = np.unique(km.labels_, return_counts=True)
    print("\nCluster labels distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} items ({count/len(km.labels_)*100:.2f}%)")
    
    # Create crosstab to analyze categorical distributions
    df_sample_cat = df.loc[sample_indices_cat].copy()
    df_sample_cat['kmodes_cluster'] = km.labels_
    
    # Analyze titleType distribution by cluster
    kmodes_type_cross = pd.crosstab(df_sample_cat['titleType'], df_sample_cat['kmodes_cluster'], normalize='columns') * 100
    
    # Visualize titleType distribution
    plt.figure(figsize=(14, 8))
    kmodes_type_cross.plot(kind='bar', stacked=True, colormap='tab20')
    plt.title('Distribution of Title Types in K-Modes Clusters')
    plt.xlabel('Title Type')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("kmodes package not installed. To use K-Modes, install with: pip install kmodes") 
except Exception as e:
    print(f"Error running K-Modes clustering: {str(e)}")

# %%
"""
## Part 3: Classification with KNN and Naive Bayes
"""

# %%
# Import additional libraries for classification
import sys
from scipy import interpolate

# Define an interp function that correctly mimics the behavior of the old scipy.interp
# This is included in your professor's methodology
def interp(x, xp, fp, left=None, right=None):
    """
    This function replicates the behavior of scipy.interp, 
    using scipy.interpolate.interp1d
    """
    # Handle edge cases
    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]
        
    # Create interpolation function
    f = interpolate.interp1d(xp, fp, bounds_error=False, fill_value=(left, right))
    
    # Return interpolated values
    return f(x)

# Make interp available in scipy namespace (following professor's approach)
sys.modules['scipy.interp'] = interp
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# %%
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Try to import scikitplot for enhanced visualizations
try:
    import scikitplot as skplt
    from scikitplot.metrics import plot_roc
    from scikitplot.metrics import plot_precision_recall
    from scikitplot.metrics import plot_cumulative_gain, plot_lift_curve
    skplt_available = True
except ImportError:
    print("scikit-plot not available. Some visualizations will be skipped.")
    skplt_available = False

# %%
"""
### Preparing Data for Classification
"""

# %%
# First, we need to prepare our target variable from the rating_numeric
# We'll create a classification problem by binning the ratings into 3 classes: low, medium, high
df['rating_class'] = pd.qcut(df['rating_numeric'], q=3, labels=[0, 1, 2])

# Display the distribution of rating classes
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rating_class')
plt.title('Distribution of Rating Classes')
plt.xlabel('Rating Class (0=Low, 1=Medium, 2=High)')
plt.ylabel('Count')
plt.grid(True, axis='y')
plt.show()

# %%
# Select features for classification
# We'll use numeric features and avoid using features that might cause data leakage
feature_cols = ['startYear_numeric', 'runtimeMinutes_numeric', 'awardWins', 'numVotes', 
                'totalImages', 'totalCredits', 'criticReviewsTotal',
                'awardNominationsExcludeWins', 'numRegions', 'userReviewsTotal', 'ratingCount',
                'canHaveEpisodes', 'isRatable', 'isAdult', 'titleType_Val', 'genre_Val', 'country_Val']

# Ensure all columns exist
feature_cols = [col for col in feature_cols if col in df.columns]
print(f"Using {len(feature_cols)} features for classification: {feature_cols}")

# Prepare feature matrix X and target vector y
X = df[feature_cols].values
y = df['rating_class'].values

# Check for any remaining NaNs in X and replace with 0
if np.isnan(X).any():
    print("Warning: X contains NaN values. Replacing with 0...")
    X = np.nan_to_num(X)

# %%
"""
### Partitioning the Data
"""

# %%
# Using the same random state as your professor's code
random_state = 0

# %%
# First split without stratification to demonstrate the difference (following professor's approach)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state
)

# %%
# Show class distribution without stratify
print("Class distribution without stratification:")
print("Original:", np.unique(y, return_counts=True)[1] / len(y))
print("Training:", np.unique(y_train, return_counts=True)[1] / len(y_train))
print("Testing:", np.unique(y_test, return_counts=True)[1] / len(y_test))

# %%
# Now use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=random_state
)

# %%
# Show class distribution with stratify
print("Class distribution with stratification:")
print("Original:", np.unique(y, return_counts=True)[1] / len(y))
print("Training:", np.unique(y_train, return_counts=True)[1] / len(y_train))
print("Testing:", np.unique(y_test, return_counts=True)[1] / len(y_test))

# %%
print(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# %%
"""
### Normalizing the Data
"""

# %%
# Using StandardScaler as in the professor's code
print("Applying StandardScaler normalization...")
norm = StandardScaler()
norm.fit(X_train)

X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

# Let's also demonstrate the RobustScaler which is less sensitive to outliers (alternative approach)
print("Applying RobustScaler normalization (alternative approach)...")
robust_norm = RobustScaler()
robust_norm.fit(X_train)

X_train_robust = robust_norm.transform(X_train)
X_test_robust = robust_norm.transform(X_test)

# Verify no NaNs in normalized data
if np.isnan(X_train_norm).any() or np.isnan(X_test_norm).any():
    print("Warning: Normalized data contains NaN values. Replacing with 0...")
    X_train_norm = np.nan_to_num(X_train_norm)
    X_test_norm = np.nan_to_num(X_test_norm)

# %%
"""
### KNN Classification
"""

# %%
# Initialize KNN classifier with same parameters as professor's approach
print("Training KNN classifier...")
try:
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
    clf.fit(X_train_norm, y_train)
    print("KNN model fitted successfully")
except Exception as e:
    print(f"Error fitting KNN model: {e}")
    # Implement alternative approach if needed
    print("Trying alternative approach with fewer neighbors...")
    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="uniform")
    clf.fit(X_train_norm, y_train)

# %%
# predict: Predict the class labels for the provided data.
y_test_pred = clf.predict(X_test_norm)
print("First 10 predictions:", y_test_pred[:10])
print("First 10 actual values:", y_test[:10])

# %%
# Calculate accuracy using accuracy_score
print("Accuracy:", accuracy_score(y_test, y_test_pred))

# %%
# Using the model's score method (same as accuracy)
try:
    print("Model score:", clf.score(X_test_norm, y_test))
except Exception as e:
    print(f"Error in scoring: {e}")

# %%
# Demonstrating that KNeighborsClassifier.score is the same as accuracy calculation
print("Manual accuracy calculation:", (y_test_pred == y_test).sum() / len(y_test))

# %%
"""
#### Performance evaluation
"""

# %%
# F1 score with different averaging methods (following professor's approach)
print("F1 (macro):", f1_score(y_test, y_test_pred, average="macro"))
print("F1 (micro):", f1_score(y_test, y_test_pred, average="micro"))

# F1 score for specific labels
for label in range(3):  # For classes 0, 1, 2
    print(f"F1 for class {label} (micro):", f1_score(y_test, y_test_pred, labels=[label], average="micro"))
    print(f"F1 for class {label} (macro):", f1_score(y_test, y_test_pred, labels=[label], average="macro"))

# %%
# Detailed classification report
print(classification_report(y_test, y_test_pred))

# %%
# Confusion matrix visualization
cf = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cf, annot=True, cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for KNN")
plt.show()

# %%
# Return probability estimates for the test data
try:
    y_test_pred_proba = clf.predict_proba(X_test_norm)
    print("First 10 probability predictions:")
    print(y_test_pred_proba[:10])
except Exception as e:
    print(f"Error in probability prediction: {e}")
    y_test_pred_proba = None

# %%
# Plot ROC curve if probability predictions are available
if y_test_pred_proba is not None and skplt_available:
    plot_roc(y_test, y_test_pred_proba)
    plt.title("ROC Curve for KNN")
    plt.show()
    
    # Calculate ROC AUC Score
    try:
        roc_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class="ovr", average="macro")
        print("ROC AUC Score:", roc_auc)
    except Exception as e:
        print(f"Error calculating ROC AUC score: {e}")

# %%
# Plot precision-recall curve if available
if y_test_pred_proba is not None and skplt_available:
    try:
        plot_precision_recall(y_test, y_test_pred_proba)
        plt.title("Precision-Recall Curve for KNN")
        plt.show()
    except Exception as e:
        print(f"Error plotting precision-recall curve: {e}")

# %%
"""
#### Repeated Holdout
"""

# %%
# Implement the repeated holdout procedure from your professor's methodology
N = 50
err = 0

try:
    for i in range(N):
        # stratified holdout
        X_rh_train, X_rh_test, y_rh_train, y_rh_test = train_test_split(X, y, test_size=0.4, stratify=y)
        
        # normalize train set
        norm.fit(X_rh_train)
        X_rh_train_norm = norm.transform(X_rh_train)
        X_rh_test_norm = norm.transform(X_rh_test)
        
        # Replace any NaNs that might have been introduced
        X_rh_train_norm = np.nan_to_num(X_rh_train_norm)
        X_rh_test_norm = np.nan_to_num(X_rh_test_norm)

        # initialize and fit classifier
        clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
        clf.fit(X_rh_train_norm, y_rh_train)

        # computing error
        acc = clf.score(X_rh_test_norm, y_rh_test)
        err += 1 - acc

    print("Overall error estimate:", err/N)
except Exception as e:
    print(f"Error in repeated holdout: {e}")

# %%
"""
#### Cross-validation
https://scikit-learn.org/stable/modules/cross_validation.html
"""

# %%
k = 10  # Number of folds

# %%
# initialize classifier
clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")

try:
    scores = cross_val_score(clf, X_train_norm, y_train, cv=k)
    print("Cross-validation scores:", scores)
except Exception as e:
    print(f"Error in cross-validation: {e}")
    scores = None

# %%
if scores is not None:
    print("Overall error estimate:", 1 - scores.mean())

# %%
if scores is not None:
    print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std()))

# %%
# Cross-validation with different scoring metrics (as in professor's code)
try:
    f1_scores = cross_val_score(clf, X_train_norm, y_train, cv=k, scoring='f1_macro')
    print("F1 macro scores:", f1_scores)
    print(f"Mean F1 macro: {f1_scores.mean():.4f}")
except Exception as e:
    print(f"Error in F1 macro cross-validation: {e}")

# %%
"""
### Hyperparameters Tuning
"""

# %%
# Testing different n_neighbors values directly (following professor's approach)
n_neighbors = range(1, 20)
scores = list()

try:
    for n in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=n, metric="euclidean", weights="uniform")
        clf.fit(X_train_norm, y_train)
        scores.append(clf.score(X_test_norm, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xticks(range(len(n_neighbors)), n_neighbors)
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.title("Effect of n_neighbors on accuracy")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error in hyperparameter tuning: {e}")

# %%
# Cross-validated parameter tuning with error bars (professor's approach)
n_neighbors = range(1, 20)
avg_scores = list()
std_scores = list()

try:
    for n in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=n, metric="euclidean", weights="uniform")
        cv_scores = cross_val_score(clf, X_train_norm, y_train, cv=k)
        avg_scores.append(np.mean(cv_scores))
        std_scores.append(np.std(cv_scores))

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(n_neighbors)), y=avg_scores, yerr=std_scores, marker='o')
    plt.xticks(range(len(n_neighbors)), n_neighbors)
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.title("Cross-validated performance with error bars")
    plt.grid(True)
    plt.show()
    
    # Find the best n_neighbors value
    best_n = n_neighbors[np.argmax(avg_scores)]
    print(f"Best n_neighbors value: {best_n} with accuracy: {max(avg_scores):.4f}")
except Exception as e:
    print(f"Error in cross-validated hyperparameter tuning: {e}")
    best_n = 5  # Default value

# %%
# Try best n_neighbors
try:
    # Use the best n_neighbors found (or default to 5)
    best_n_neighbors = best_n if 'best_n' in locals() else 5
    
    clf = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric="euclidean", weights="uniform")
    clf.fit(X_train_norm, y_train)
    y_test_pred = clf.predict(X_test_norm)
    print(f"Accuracy with n_neighbors={best_n_neighbors}:", accuracy_score(y_test, y_test_pred))
except Exception as e:
    print(f"Error in optimal model evaluation: {e}")

# %%
"""
#### Grid Search
"""

# %%
try:
    # Define parameter grid
    param_grid = {
        "n_neighbors": np.arange(1, min(30, X_train.shape[0]//10)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "cityblock"],
    }

    print("Starting grid search...")
    
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid=param_grid,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0),
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(X_train_norm, y_train)
    clf = grid.best_estimator_
    
    print("Grid search completed.")
    print("Best parameters:", grid.best_params_, "Best score:", grid.best_score_)
except Exception as e:
    print(f"Error in grid search: {e}")
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
    clf.fit(X_train_norm, y_train)

# %%
try:
    y_test_pred = clf.predict(X_test_norm)
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
except Exception as e:
    print(f"Error in best model evaluation: {e}")

# %%
try:
    print("Model score:", clf.score(X_test_norm, y_test))
except Exception as e:
    print(f"Error in model scoring: {e}")

# %%
try:
    results = pd.DataFrame(grid.cv_results_)
    
    # Creating metric_weight column for better visualization (as in professor's code)
    results["metric_weight"] = results["param_metric"] + ", " + results["param_weights"]
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results, x="param_n_neighbors", y="mean_test_score", hue="metric_weight"
    )
    plt.title("Grid Search Results")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Mean test score")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error plotting grid search results: {e}")
    results = None

# %%
"""
### Naive Bayes
"""

# %%
"""
#### Gaussian
"""

# %%
try:
    clf_nb = GaussianNB()
    
    print("Fitting Gaussian Naive Bayes model...")
    clf_nb.fit(X_train, y_train)
    print("Model fitted successfully.")
except Exception as e:
    print(f"Error fitting Gaussian NB model: {e}")

# %%
try:
    y_pred = clf_nb.predict(X_test)
    print("First 10 predictions:", y_pred[:10])
except Exception as e:
    print(f"Error in Gaussian NB prediction: {e}")
    y_pred = None

# %%
if y_pred is not None:
    print(classification_report(y_test, y_pred))

# %%
try:
    y_test_proba_nb = clf_nb.predict_proba(X_test)
    print("First 5 probability predictions:", y_test_proba_nb[:5])
except Exception as e:
    print(f"Error in Gaussian NB probability prediction: {e}")
    y_test_proba_nb = None

# %%
if y_test_proba_nb is not None and skplt_available:
    try:
        plot_roc(y_test, y_test_proba_nb)
        plt.title("ROC Curve for Gaussian Naive Bayes")
        plt.show()
        
        roc_auc_nb = roc_auc_score(y_test, y_test_proba_nb, multi_class="ovr", average="macro")
        print(f"ROC AUC Score: {roc_auc_nb:.4f}")
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

# %%
"""
### Binary Classification Example
"""

# %%
# plot_cumulative_gain and plot_lift_curve only work in a binary classification case
# Convert our multiclass problem to binary (high rating vs low rating)
try:
    # Keep only classes 0 and 2 (dropping the middle class) - exactly as in professor's approach
    binary_indices = np.where((y == 0) | (y == 2))[0]
    X_binary = X[binary_indices]
    y_binary = y[binary_indices]
    # Map class 2 to 1 to get a binary target (0 and 1)
    y_binary = np.where(y_binary == 2, 1, 0)
    
    print(f"Binary dataset shape: X={X_binary.shape}, y={y_binary.shape}")
    print(f"Binary class distribution: {np.unique(y_binary, return_counts=True)}")
except Exception as e:
    print(f"Error creating binary dataset: {e}")

# %%
try:
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_binary, y_binary, test_size=0.3, stratify=y_binary, random_state=random_state
    )
    print(f"Binary train/test shapes: X_train={X_train_bin.shape}, X_test={X_test_bin.shape}")
except Exception as e:
    print(f"Error splitting binary dataset: {e}")

# %%
try:
    clf_bin = GaussianNB()
    clf_bin.fit(X_train_bin, y_train_bin)
    print("Binary classifier fitted successfully")
except Exception as e:
    print(f"Error fitting binary classifier: {e}")

# %%
try:
    y_test_pred_proba = clf_bin.predict_proba(X_test_bin)
    print("First 5 probability predictions:", y_test_pred_proba[:5])
except Exception as e:
    print(f"Error in binary classifier probability prediction: {e}")
    y_test_pred_proba = None

# %%
if y_test_pred_proba is not None and skplt_available:
    try:
        plot_roc(y_test_bin, y_test_pred_proba)
        plt.title("ROC Curve for Binary Classification")
        plt.show()
    except Exception as e:
        print(f"Error plotting binary ROC curve: {e}")

# %%
if y_test_pred_proba is not None and skplt_available:
    try:
        plot_precision_recall(y_test_bin, y_test_pred_proba)
        plt.title("Precision-Recall Curve for Binary Classification")
        plt.show()
    except Exception as e:
        print(f"Error plotting precision-recall curve: {e}")

# %%
# Cumulative gain and lift curves (binary classification only)
if y_test_pred_proba is not None and skplt_available:
    try:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
        plot_cumulative_gain(y_test_bin, y_test_pred_proba, ax=axs[0])
        axs[0].set_title("Cumulative Gain Curve")
        
        plot_lift_curve(y_test_bin, y_test_pred_proba, ax=axs[1])
        axs[1].set_title("Lift Curve")
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting cumulative gain and lift curves: {e}")

# %%
# Also implement binary classification with KNN (to compare the results)
try:
    # Normalize the data
    norm_bin = StandardScaler()
    X_train_bin_norm = norm_bin.fit_transform(X_train_bin)
    X_test_bin_norm = norm_bin.transform(X_test_bin)
    
    # KNN
    clf_bin_knn = KNeighborsClassifier(n_neighbors=5)
    clf_bin_knn.fit(X_train_bin_norm, y_train_bin)
    y_pred_bin_knn = clf_bin_knn.predict(X_test_bin_norm)
    
    print("\nBinary Classification Results:")
    print(f"KNN Accuracy: {accuracy_score(y_test_bin, y_pred_bin_knn):.4f}")
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test_bin, clf_bin.predict(X_test_bin)):.4f}")
except Exception as e:
    print(f"Error in KNN binary classification: {e}")

# %%
"""
## Part 4: Regression Analysis
"""

# %%
# Import necessary libraries for regression (some may already be imported)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# %%
"""
### Preparing Data for Regression
"""

# %%
# Create a separate copy of the data for regression
df_reg = df.copy()

# Convert rating from categorical to numeric if it's not already
# Using the bin_to_midpoint function from the regression pattern file
def bin_to_midpoint(rating_str):
    try:
        # Extract the two values inside the interval notation
        left, right = rating_str.strip('()[]').split(', ')
        # Calculate the midpoint
        return (float(left) + float(right)) / 2
    except:
        return np.nan

# Apply if needed - our dataset already has rating_numeric
if 'rating_numeric' not in df_reg.columns:
    df_reg['rating_numeric'] = df_reg['rating'].apply(bin_to_midpoint)

# Handle any remaining missing values for regression
for col in ['startYear_numeric', 'runtimeMinutes_numeric', 'awardWins', 'numVotes', 
            'totalImages', 'totalCredits', 'criticReviewsTotal',
            'awardNominationsExcludeWins', 'numRegions', 'userReviewsTotal', 'ratingCount']:
    if col in df_reg.columns and df_reg[col].isnull().sum() > 0:
        print(f"Filling {df_reg[col].isnull().sum()} missing values in {col}")
        median_val = df_reg[col].median()
        df_reg[col] = df_reg[col].fillna(median_val)

# Remove rows with missing target values (rating_numeric)
df_reg = df_reg.dropna(subset=['rating_numeric'])
print(f"Regression dataset shape after handling missing values: {df_reg.shape}")

# %%
# Split the data into train and test sets
df_train, df_test = train_test_split(df_reg, test_size=0.3, random_state=100)
print(f"Training set: {df_train.shape}, Test set: {df_test.shape}")

# %%
"""
### Simple Regression: Predicting Rating from Number of Votes
"""

# %%
# Prepare data for simple regression
x_train = df_train["numVotes"].values.reshape(-1, 1)
y_train = df_train["rating_numeric"].values

x_test = df_test["numVotes"].values.reshape(-1, 1)
y_test = df_test["rating_numeric"].values

# %%
"""
#### Linear Regression
"""

# %%
reg = LinearRegression()
reg.fit(x_train, y_train)

# %%
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# %%
# Visualize the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x="numVotes", y="rating_numeric", alpha=0.5)
plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")
plt.xscale('log')  # Use log scale for better visualization since numVotes can have large range
plt.title('Linear Regression: Number of Votes vs Rating')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
# Evaluate the model
y_pred = reg.predict(x_test)

print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
# Visualize predictions on test data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_test, x="numVotes", y="rating_numeric", alpha=0.5)
plt.plot(x_test, reg.coef_[0]*x_test+reg.intercept_, c="red")
plt.xscale('log')  # Use log scale
plt.title('Linear Regression: Test Data')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### Ridge Regression (Regularized)
"""

# %%
# For regularized models, scaling features is important
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# %%
# Try Ridge regression with alpha=1.0
reg = Ridge(alpha=1.0)  # alpha is the regularization strength
reg.fit(x_train_scaled, y_train)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# %%
# Visualize Ridge predictions
y_pred_train = reg.predict(x_train_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x="numVotes", y="rating_numeric", alpha=0.5)
# Sort x_train for proper line plotting with transformed data
sorted_indices = np.argsort(x_train.flatten())
plt.scatter(x_train[sorted_indices], y_pred_train[sorted_indices], color='red', s=15)
plt.xscale('log')
plt.title('Ridge Regression: Training Data')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
# Evaluate Ridge model
y_pred = reg.predict(x_test_scaled)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
"""
#### Lasso Regression (Regularized)
"""

# %%
# Try Lasso regression with alpha=0.1
reg = Lasso(alpha=0.1)  # alpha is the regularization strength
reg.fit(x_train_scaled, y_train)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# %%
# Visualize Lasso predictions
y_pred_train = reg.predict(x_train_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x="numVotes", y="rating_numeric", alpha=0.5)
# Sort x_train for proper line plotting with transformed data
sorted_indices = np.argsort(x_train.flatten())
plt.scatter(x_train[sorted_indices], y_pred_train[sorted_indices], color='red', s=15)
plt.xscale('log')
plt.title('Lasso Regression: Training Data')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
# Evaluate Lasso model
y_pred = reg.predict(x_test_scaled)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
"""
### Nonlinear Regression Models
"""

# %%
"""
#### Decision Tree Regressor
"""

# %%
# Use Decision Tree for regression
reg = DecisionTreeRegressor(max_depth=5)  # Limit depth to prevent overfitting
reg.fit(x_train, y_train)  # Decision trees don't require scaling

# %%
# Evaluate Decision Tree model
y_pred = reg.predict(x_test)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
# Visualize Decision Tree predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x="numVotes", y="rating_numeric", label="True", alpha=0.5)
sns.scatterplot(data=df_train, x="numVotes", y=reg.predict(x_train), label="Predicted", marker="X", alpha=0.7)
plt.xscale('log')
plt.legend()
plt.title('Decision Tree Regressor: Training Data')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### KNN Regressor
"""

# %%
# KNN for regression
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(x_train_scaled, y_train)  # Use scaled features for KNN

# %%
# Evaluate KNN model
y_pred = reg.predict(x_test_scaled)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
# Visualize KNN predictions
y_pred_train = reg.predict(x_train_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x="numVotes", y="rating_numeric", label="True", alpha=0.5)
plt.scatter(x_train, y_pred_train, label="Predicted", marker="X", alpha=0.7)
plt.xscale('log')
plt.legend()
plt.title('KNN Regressor: Training Data')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
### Multiple Regression: Using Multiple Features to Predict Rating
"""

# %%
# Choose numeric features for multiple regression
features = ["numVotes", "runtimeMinutes_numeric", "totalImages", "totalCredits", "awardWins"]

# Ensure all features exist in the dataframe
features = [f for f in features if f in df_train.columns]
print(f"Using features for multiple regression: {features}")

X_train = df_train[features].values
y_train = df_train["rating_numeric"].values

X_test = df_test[features].values
y_test = df_test["rating_numeric"].values

# Scale all features for the multiple regression models
scaler_multi = StandardScaler()
X_train_scaled = scaler_multi.fit_transform(X_train)
X_test_scaled = scaler_multi.transform(X_test)

# %%
"""
#### Multiple Linear Regression
"""

# %%
reg = LinearRegression()
reg.fit(X_train, y_train)  # Linear regression doesn't require scaling

# %%
# Evaluate the multiple regression model
y_pred = reg.predict(X_test)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))
print('\nFeature Coefficients:')
for feature, coef in zip(features, reg.coef_):
    print(f"{feature}: {coef:.6f}")

# %%
# Since we have multiple features, we'll visualize one at a time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_test, x="totalImages", y="rating_numeric", label="True", alpha=0.5)
sns.scatterplot(data=df_test, x="totalImages", y=reg.predict(X_test), label="Predicted", marker="X", alpha=0.7)
plt.legend()
plt.title('Multiple Linear Regression: Test Data')
plt.xlabel('Total Images')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
#### Multiple Nonlinear Regression: Decision Tree
"""

# %%
reg = DecisionTreeRegressor(max_depth=6)
reg.fit(X_train, y_train)

# %%
# Evaluate the decision tree model
y_pred = reg.predict(X_test)
print('R2: %.3f' % r2_score(y_test, y_pred))
print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

# %%
# Visualizing feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': reg.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Decision Tree Feature Importance')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Visualization with one of the features
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_test, x="totalImages", y="rating_numeric", label="True", alpha=0.5)
sns.scatterplot(data=df_test, x="totalImages", y=reg.predict(X_test), label="Predicted", marker="X", alpha=0.7)
plt.legend()
plt.title('Decision Tree Regressor: Test Data')
plt.xlabel('Total Images')
plt.ylabel('Rating')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
### Multivariate Regression (2+ target variables)
"""

# %%
# Features for multivariate regression
multi_features = ["numVotes", "runtimeMinutes_numeric", "awardWins"]
multi_features = [f for f in multi_features if f in df_train.columns]
print(f"Using features for multivariate regression: {multi_features}")

multi_targets = ["rating_numeric", "totalImages"]
multi_targets = [t for t in multi_targets if t in df_train.columns]
print(f"Predicting targets: {multi_targets}")

X_train_multi = df_train[multi_features].values
y_train_multi = df_train[multi_targets].values

X_test_multi = df_test[multi_features].values
y_test_multi = df_test[multi_targets].values

# %%
# Using Decision Tree for multivariate regression
reg_multi = DecisionTreeRegressor(max_depth=6)
reg_multi.fit(X_train_multi, y_train_multi)

# %%
# Evaluate the multivariate model
y_pred_multi = reg_multi.predict(X_test_multi)
print('R2: %.3f' % r2_score(y_test_multi, y_pred_multi))
print('MSE: %.3f' % mean_squared_error(y_test_multi, y_pred_multi))
print('MAE: %.3f' % mean_absolute_error(y_test_multi, y_pred_multi))

# %%
# Visualization for the first target (rating)
if "totalVideos" in df_test.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_test, x="totalVideos", y="rating_numeric", label="True", alpha=0.5)
    sns.scatterplot(data=df_test, x="totalVideos", y=reg_multi.predict(X_test_multi)[:, 0], label="Predicted", marker="X", alpha=0.7)
    plt.legend()
    plt.title('Multivariate Regression: Predicting Rating')
    plt.xlabel('Total Videos')
    plt.ylabel('Rating')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Visualization for the second target (totalImages)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_test, x="totalVideos", y="totalImages", label="True", alpha=0.5)
    sns.scatterplot(data=df_test, x="totalVideos", y=reg_multi.predict(X_test_multi)[:, 1], label="Predicted", marker="X", alpha=0.7)
    plt.legend()
    plt.title('Multivariate Regression: Predicting Total Images')
    plt.xlabel('Total Videos')
    plt.ylabel('Total Images')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# %%
"""
## Part 5: Pattern Mining
"""

# %%
# Import necessary libraries for pattern mining
try:
    from fim import apriori, fpgrowth
    print("Successfully imported fim package for pattern mining")
except ImportError:
    print("Warning: fim package not installed. Pattern mining section will not work.")
    print("To install, run: pip install pyfim")

# %%
"""
### Feature preprocessing for pattern mining
"""

# %%
# Reset to original dataframe for pattern mining
try:
    # Create a copy for pattern mining
    df_pattern = df.copy()
    
    # Use existing bin_to_midpoint function if needed
    if 'rating_numeric' not in df_pattern.columns:
        print("Generating rating_numeric from rating")
        df_pattern['rating_numeric'] = df_pattern['rating'].apply(bin_to_midpoint)
    
    # remove columns not needed for pattern mining
    columns_to_drop = ["originalTitle", "endYear", "worstRating", "bestRating", "countryOfOrigin"]
    df_pattern.drop([col for col in columns_to_drop if col in df_pattern.columns], axis=1, inplace=True)
    
    print(f"Initial pattern mining dataset shape: {df_pattern.shape}")
except Exception as e:
    print(f"Error preparing data for pattern mining: {e}")

# %%
"""
### Pattern mining preprocessing
"""

# %%
try:
    # Discretize continuous variables into bins for pattern mining
    print("Discretizing numeric variables for pattern mining")
    
    # Convert rating to categorical bins
    df_pattern["RatingBin"] = pd.cut(df_pattern["rating_numeric"].astype(float), 
                                    bins=[0, 3, 5, 7, 9, 10], 
                                    labels=["Very Low", "Low", "Medium", "High", "Very High"])
    
    # Convert numVotes to bins
    if "numVotes" in df_pattern.columns:
        df_pattern["VotesBin"] = pd.cut(pd.to_numeric(df_pattern["numVotes"], errors='coerce'),
                                        bins=[0, 100, 1000, 10000, 100000, float('inf')],
                                        labels=["Few", "Some", "Many", "Lots", "Massive"])
    
    # Convert runtimeMinutes to bins
    if "runtimeMinutes_numeric" in df_pattern.columns:
        df_pattern["RuntimeBin"] = pd.cut(pd.to_numeric(df_pattern["runtimeMinutes_numeric"], errors='coerce'),
                                        bins=[0, 30, 60, 90, 120, float('inf')],
                                        labels=["Very Short", "Short", "Medium", "Long", "Very Long"])
    
    # Convert startYear to decades
    if "startYear_numeric" in df_pattern.columns:
        df_pattern["DecadeBin"] = pd.cut(df_pattern["startYear_numeric"],
                                        bins=[1900, 1950, 1970, 1990, 2000, 2010, 2020, 2030],
                                        labels=["Pre-1950s", "50s-60s", "70s-80s", "90s", "2000s", "2010s", "2020s"])
    
    print("Checking discretized data")
    binned_cols = [col for col in ["RatingBin", "VotesBin", "RuntimeBin", "DecadeBin"] if col in df_pattern.columns]
    if binned_cols:
        print(df_pattern[binned_cols].head())
        
    # Drop the original continuous variables
    continuous_cols = ["numVotes", "runtimeMinutes_numeric", "startYear_numeric", "rating_numeric"]
    df_pattern.drop([col for col in continuous_cols if col in df_pattern.columns], axis=1, inplace=True)
    
    # Process categorical variables
    print("Processing categorical variables")
    
    # Add titleType as a categorical feature
    if "titleType" in df_pattern.columns:
        df_pattern["titleType"] = df_pattern["titleType"].fillna("Unknown") + "_Type"
    
    # Process genres to get the primary genre
    def get_primary_genre(genre_str):
        if pd.isna(genre_str) or genre_str == '':
            return "Unknown_Genre"
        try:
            # Try to extract the first genre from the string
            return genre_str.split(',')[0] + "_Genre"
        except:
            return "Unknown_Genre"
    
    # Apply if genres column exists
    if "genres" in df_pattern.columns:
        df_pattern["PrimaryGenre"] = df_pattern["genres"].apply(get_primary_genre)
    
    # Convert boolean columns to categorical strings
    for col in ['canHaveEpisodes', 'isRatable', 'isAdult']:
        if col in df_pattern.columns:
            df_pattern[col] = df_pattern[col].map({'TRUE': f"Yes_{col}", 'FALSE': f"No_{col}"})
            df_pattern[col] = df_pattern[col].fillna(f"Unknown_{col}")
    
    # Handle NaN values in categorical columns
    for col in ['RatingBin', 'VotesBin', 'RuntimeBin', 'DecadeBin']:
        if col in df_pattern.columns and df_pattern[col].isna().any():
            # Convert to string type first to allow adding new values
            df_pattern[col] = df_pattern[col].astype(str)
            # Now we can replace NaN values
            df_pattern[col] = df_pattern[col].replace('nan', f'Unknown_{col}')
    
    print("Final pattern mining dataset preview:")
    print(df_pattern.head())
    
    # Prepare data for pattern mining - select relevant columns
    pattern_cols = []
    for col in ['RatingBin', 'VotesBin', 'RuntimeBin', 'DecadeBin', 'titleType', 'PrimaryGenre']:
        if col in df_pattern.columns:
            pattern_cols.append(col)
    
    # Add boolean columns if they exist
    for col in ['canHaveEpisodes', 'isRatable', 'isAdult']:
        if col in df_pattern.columns:
            pattern_cols.append(col)
    
    # No need to drop NaN rows - we've handled them appropriately
    df_pattern = df_pattern[pattern_cols]
    print(f"Pattern mining dataset shape: {df_pattern.shape}")
    
    # Convert dataframe to list of transactions
    transactions = []
    
    # Handle NaN values by converting each row separately
    for _, row in df_pattern.iterrows():
        transaction = []
        for item in row:
            if pd.isna(item):
                transaction.append("Unknown")
            else:
                transaction.append(str(item))
        transactions.append(transaction)
    
    # Preview the first few transactions
    print("Sample transactions after cleaning:")
    for i, transaction in enumerate(transactions[:3]):
        print(f"Transaction {i+1}: {transaction}")

except Exception as e:
    print(f"Error in pattern mining preprocessing: {e}")

# %%
"""
### Apriori Algorithm
"""

# %%
try:
    # Frequent itemset mining
    supp = 20  # 20% support threshold
    zmin = 2   # minimum number of items per itemset
    
    print("Mining frequent itemsets with Apriori algorithm...")
    itemsets = apriori(transactions, target="s", supp=supp, zmin=zmin, report="S")
    itemsets_df = pd.DataFrame(itemsets, columns=["frequent_itemset", "support"])
    print(f"Found {len(itemsets)} frequent itemsets")
    print(itemsets_df.head(10))  # Show the first 10 frequent itemsets
    
    # Mining closed itemsets
    print("\nMining closed itemsets...")
    itemsets = apriori(transactions, target="c", supp=supp, zmin=zmin, report="S")
    closed_df = pd.DataFrame(itemsets, columns=["closed_itemset", "support"])
    print(f"Found {len(itemsets)} closed itemsets")
    print(closed_df.head(10))  # Show the first 10 closed itemsets
    
    # Mining maximal itemsets
    print("\nMining maximal itemsets...")
    itemsets = apriori(transactions, target="m", supp=supp, zmin=zmin, report="S")
    maximal_df = pd.DataFrame(itemsets, columns=["maximal_itemset", "support"])
    print(f"Found {len(itemsets)} maximal itemsets")
    print(maximal_df.head(10))  # Show the first 10 maximal itemsets
except NameError:
    print("Apriori algorithm not available. Please install pyfim package.")
except Exception as e:
    print(f"Error in Apriori algorithm: {e}")

# %%
"""
### Support Analysis
"""

# %%
try:
    # Analyze how the number of itemsets changes with support
    len_max_it = []
    len_cl_it = []
    max_supp = 25
    
    for i in range(2, max_supp):
        max_itemsets = apriori(transactions, target="m", supp=i, zmin=zmin)
        cl_itemsets = apriori(transactions, target="c", supp=i, zmin=zmin)
        len_max_it.append(len(max_itemsets))
        len_cl_it.append(len(cl_itemsets))
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(2, max_supp), len_max_it, label="maximal")
    plt.plot(np.arange(2, max_supp), len_cl_it, label="closed")
    plt.legend()
    plt.xlabel("%support")
    plt.ylabel("number of itemsets")
    plt.title("Support vs Number of Itemsets")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Analyze support vs number of itemsets for high/low ratings
    filter_1 = []
    filter_2 = []
    for i in range(2, max_supp):
        max_itemsets = apriori(transactions, target="m", supp=i, zmin=zmin)
        filter_1.append(len([item for item in max_itemsets if "High" in item[0]]))
        filter_2.append(len([item for item in max_itemsets if "Low" in item[0]]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(2, max_supp), filter_1, label="w/ High Rating")
    plt.plot(np.arange(2, max_supp), filter_2, label="w/ Low Rating")
    plt.legend()
    plt.xlabel("%support")
    plt.ylabel("number of itemsets")
    plt.title("Support vs Number of Itemsets with High/Low Ratings")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
except NameError:
    print("Support analysis requires apriori algorithm. Please install pyfim package.")
except Exception as e:
    print(f"Error in support analysis: {e}")

# %%
"""
### Association Rules
"""

# %%
try:
    # Mine association rules
    conf = 60  # 60% confidence threshold
    rules = apriori(transactions, target="r", supp=supp, zmin=zmin, conf=conf, report="aScl")
    rules_df = pd.DataFrame(
        rules,
        columns=[
            "consequent",
            "antecedent",
            "abs_support",
            "%_support",
            "confidence",
            "lift",
        ],
    )
    print("Top 10 rules by lift:")
    print(rules_df.sort_values(by="lift", axis=0, ascending=False).head(10))
    
    # Analyze rules that predict rating levels
    high_rating_rules = rules_df[rules_df["consequent"] == "High"]
    print("\nTop 10 rules predicting high ratings:")
    print(high_rating_rules.sort_values(by="lift", ascending=False).head(10))
    
    # Look at rules with highest confidence for high ratings
    if not high_rating_rules.empty:
        best_rule = high_rating_rules.sort_values(by="confidence", ascending=False).iloc[0]
        print("\nBest rule for predicting high ratings:")
        print("To predict:", best_rule.iloc[0])
        print("Using rule:", best_rule.iloc[1])
        print("With confidence:", best_rule.iloc[4])
        print("And lift:", best_rule.iloc[5])
    else:
        print("No rules found with 'High' as consequent. Try lowering the confidence or support threshold.")
        
    # Analyze the trade-off between support and confidence
    len_r = []
    min_sup = 1
    max_sup = 20
    min_conf = 50
    max_conf = 90
    
    for i in range(min_sup, max_sup):  # support
        len_r_wrt_i = []
        for j in range(min_conf, max_conf):  # confidence
            rules = apriori(transactions, target="r", supp=i, zmin=zmin, conf=j, report="aScl")
            len_r_wrt_i.append(len(rules))
        len_r.append(len_r_wrt_i)
    
    len_r = np.array(len_r)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(len_r, cmap="Blues", fmt='g')
    plt.yticks(np.arange(0, max_sup-min_sup +1, 5), np.arange(min_sup, max_sup+1, 5))
    plt.xticks(np.arange(0, max_conf-min_conf+1, 5), np.arange(min_conf, max_conf+1, 5))
    plt.xlabel("%confidence")
    plt.ylabel("%support")
    plt.title("Number of Rules by Support and Confidence")
    plt.show()
except NameError:
    print("Association rule mining requires apriori algorithm. Please install pyfim package.")
except Exception as e:
    print(f"Error in association rule mining: {e}")

# %%
"""
### FP-Growth Algorithm
"""

# %%
try:
    # Mine frequent itemsets using FP-Growth
    supp = 20  # 20% support threshold
    zmin = 2   # minimum number of items per itemset
    
    print("Mining frequent itemsets with FP-Growth algorithm...")
    itemsets = fpgrowth(transactions, target="s", supp=supp, zmin=zmin, report="S")
    fp_itemsets_df = pd.DataFrame(itemsets, columns=["frequent_itemset", "support"])
    print(f"Found {len(itemsets)} frequent itemsets")
    print(fp_itemsets_df.head(10))
    
    # Mine association rules using FP-Growth
    conf = 70  # 70% confidence threshold
    rules = fpgrowth(transactions, target="r", supp=supp, zmin=zmin, conf=conf, report="aScl")
    fp_rules_df = pd.DataFrame(
        rules,
        columns=[
            "consequent",
            "antecedent",
            "abs_support",
            "%_support",
            "confidence",
            "lift",
        ],
    )
    print("\nTop 10 rules by lift using FP-Growth:")
    print(fp_rules_df.sort_values(by="lift", ascending=False).head(10))
except NameError:
    print("FP-Growth algorithm not available. Please install pyfim package.")
except Exception as e:
    print(f"Error in FP-Growth algorithm: {e}")

# %%
"""
### Compare Apriori and FP-Growth Performance
"""

# %%
try:
    import time
    
    # Measure execution time for Apriori
    start_time = time.time()
    apriori_itemsets = apriori(transactions, target="s", supp=supp, zmin=zmin, report="S")
    apriori_time = time.time() - start_time
    print(f"Apriori execution time: {apriori_time:.4f} seconds")
    print(f"Found {len(apriori_itemsets)} itemsets")
    
    # Measure execution time for FP-Growth
    start_time = time.time()
    fpgrowth_itemsets = fpgrowth(transactions, target="s", supp=supp, zmin=zmin, report="S")
    fpgrowth_time = time.time() - start_time
    print(f"FP-Growth execution time: {fpgrowth_time:.4f} seconds")
    print(f"Found {len(fpgrowth_itemsets)} itemsets")
    
    # Compare visually
    plt.figure(figsize=(10, 6))
    plt.bar(['Apriori', 'FP-Growth'], [apriori_time, fpgrowth_time])
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
except NameError:
    print("Performance comparison requires both algorithms. Please install pyfim package.")
except Exception as e:
    print(f"Error in performance comparison: {e}")

# %%
"""
### Visualize Most Interesting Rules
"""

# %%
try:
    # Extract top 10 rules by lift
    top_rules = rules_df.sort_values('lift', ascending=False).head(10)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='lift', y=top_rules.index, data=top_rules)
    plt.title('Top 10 Rules by Lift')
    plt.xlabel('Lift')
    plt.ylabel('Rule Index')
    plt.tight_layout()
    plt.show()
    
    # Create rule strength visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(rules_df['confidence'], rules_df['lift'], alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Lift')
    plt.title('Rule Strength (Confidence vs Lift)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
except NameError:
    print("Rule visualization requires successful rule mining. Please install pyfim package.")
except Exception as e:
    print(f"Error in rule visualization: {e}")

# %%