import pandas as pd
from pathlib import Path

dfm = pd.read_csv(Path('ml-25m') / 'movies.csv')
dfr = pd.read_csv(Path('ml-25m') / 'ratings.csv')

# remove users who gives extreme ratings -> rate everything highest or lowest
bools = dfr.groupby('userId').mean()['rating'].isin([5,0.5])
idx = bools[bools].index
dfr = dfr[~dfr['userId'].isin(idx)]

idx, jdx = [1], [2]
while len(idx) > 0 or len(jdx) > 0:

    # Remove users who rate less than 50 movies
    invalid_user = dfr['userId'].value_counts(ascending=True) < 50
    idx = invalid_user[invalid_user].index
    dfr = dfr[~dfr['userId'].isin(idx)]
    print("removing", len(idx), "users")

    # Remove movies that is less than 100 watch 
    invalid_movie = dfr.groupby('movieId').count()['rating'] < 100
    jdx = invalid_movie[invalid_movie].index
    dfr = dfr[~dfr['movieId'].isin(jdx)]
    print("removing", len(jdx), "movies")

print('Movie number:', len(dfr['movieId'].unique()))
print('User number:', len(dfr['userId'].unique()))
print('Rating number:', len(dfr))
print('Sparcity of the dataset is: ', len(dfr) / (len(dfr['movieId'].unique()) * len(dfr['userId'].unique())* 100), '%')