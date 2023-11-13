# Import Pandas
# 並且assign他為pd
import pandas as pd
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel



metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
#Print plot overviews of the first 5 movies. (預設5 rows)
# metadata['overview'].head()


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
# tfidf_matrix.shape

#Array mapping from feature integer indices to feature name.
tfidf.get_feature_names_out()[5000:5010]

# Compute the cosine similarity matrix
# 用來計算兩個矩陣的點積，linear_kernel是用來計算兩個vector的linear kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# cosine_sim.shape
# cosine_sim[1]

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
indices[:10]
# print(indices[:10])

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
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

print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Godfather'))


# source: https://www.datacamp.com/tutorial/recommender-systems-python

