
import pandas as pd


metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
# keywords是要轉換的物件，id是要轉換的row名稱，int是要轉換的資料類別
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
# merge()可以合併Pandas兩個資料匡
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Print the first two movies of your newly merged metadata
# print(metadata.head(2))

# Parse the stringified features into their corresponding python objects

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


# Import Numpy
import numpy as np

# 若在x裡面找不到Director這個Key，就回傳np.nan代表missing value，否則就回傳i['name']
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
	# isinstance(x, list)這個函數檢查x是否為list
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Print the new features of the first 3 films
# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)

# print(metadata[['soup']].head(2))

# Import CountVectorizer and create the count matrix
# CountVectorizer把文本轉為詞頻矩陣，每個元素表示一個詞在文本中出現次數
# TF-IDF跟CountVectorizer一樣，但是會多乘以一個「一個詞在所有文本中出現頻率的倒數」
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
# count是要使用的模型，metadata['soup']是要轉換的資料
count_matrix = count.fit_transform(metadata['soup'])

# print(count_matrix.shape)

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]
# print(get_recommendations('The Dark Knight Rises', cosine_sim2))
# print(get_recommendations('The Godfather', cosine_sim2))

# source: https://www.datacamp.com/tutorial/recommender-systems-python


