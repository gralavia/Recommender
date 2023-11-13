
# Import Pandas
# 並且assign他為pd
import pandas as pd

# Load Movies Metadata
# 用pd.read_csv() function來讀檔案
# low_memory=False是代表一次性把整個檔案讀進記憶體，而不是讀取部分來優化memory
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
# head()是用來回傳前面n row的method
metadata.head(3)
# Calculate mean of vote average column全部電影的平均票數
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
# 代表要納入chart，一部電影至少要獲得m票
# quantile() function是用來計算90th percentile，代表90%的電影得票數低於m
# 因此要找出10%的目標
m = metadata['vote_count'].quantile(0.90)
print(m)


# Filter out all qualified movies into a new DataFrame
# .copy()確保新的q_movies跟原始資料獨立開來，任何q_movies的改動都不會影響原始資料
# loc[]是用來過濾選擇'vote_count'欄位的，並且只過濾>=m的row
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
# .shape是用來取得dimension的，以這邊來說return (4555, 24)代表有4555 rows, 24 col
q_movies.shape
metadata.shape

# =====
# Function that computes the weighted rating of each movie
# v/(v+m) 的值接近於 1，這意味著R的權重較大。當投票數較少時，v/(v+m) 的值接近於 0，這意味著C的權重較大。
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
# apply()會自動把每一行data當作x傳遞給weighted_rating()函數
# axis=0 表示 apply() 函數將在每列數據上應用。(預設)
# axis=1 表示 apply() 函數將在每行數據上應用。
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 20 movies

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))

# source: https://www.datacamp.com/tutorial/recommender-systems-python


