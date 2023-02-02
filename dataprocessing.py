import pandas as pd
from pathlib import Path
import matplotlib as plt
import plotly.express as px


'''Dataset loading'''
# This might take a long time cuz the engine in python :(
dfm = pd.read_csv(Path('ml-10M100K') / 'movies.dat', delimiter='::', engine= 'python', header=None, names=['movieId', 'title', 'genres'])
dfr = pd.read_csv(Path('ml-10M100K') / 'ratings.dat', delimiter='::', engine= 'python', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])


'''Data cleaning and processing'''
# Dropping unused column to save memory
dfr = dfr.drop(columns="timestamp")

# Change rating to unsigned integer to save some storage space of the machine
dfr['rating'] = (dfr['rating']*2)
dfr = dfr.apply(pd.to_numeric, downcast="unsigned")


# remove users who gives extreme ratings -> rate everything highest or lowest
bools = dfr.groupby('userId').mean()['rating'].isin([10,1])
idx = bools[bools].index
dfr = dfr[~dfr['userId'].isin(idx)]

idx, jdx = [1], [2]
while len(idx) > 0 or len(jdx) > 0:

    # Remove users who rate less than 50 movies
    invalid_user = dfr['userId'].value_counts(ascending=True) < 50
    idx = invalid_user[invalid_user].index
    dfr = dfr[~dfr['userId'].isin(idx)]
    if __name__=="__main__":
        print("removing", len(idx), "users for they rate less than 50 movies.")

    # Remove movies that is less than 100 watch 
    invalid_movie = dfr.groupby('movieId').count()['rating'] < 100
    jdx = invalid_movie[invalid_movie].index
    dfr = dfr[~dfr['movieId'].isin(jdx)]
    if __name__=="__main__":
        print("removing", len(jdx), "movies for they having less than 100 watch.")

dfr.reset_index(inplace=True, drop = True)

# Reindex movieId and userId (start from 0 to the count number) so that they'll take less memory during one-hot stage
# movieId
movieIds = dfr['movieId'].unique()
n_movieIds = len(movieIds)
movieMap = {movieIds[i]: i for i in range(n_movieIds)}
dfm = dfm[dfm['movieId'].isin(movieIds)]
dfm['movieId'] = dfm['movieId'].map(movieMap)
dfr['movieId'] = dfr['movieId'].map(movieMap)
# userId
userIds = dfr['userId'].unique()
n_userIds = len(userIds)
userMap = {userIds[i]: i for i in range(n_userIds)}
dfr['userId'] = dfr['userId'].map(userMap)


'''Data Overview'''

if __name__=="__main__":
    print('Movie number:', len(dfr['movieId'].unique()))
    print('User number:', len(dfr['userId'].unique()))
    print('Movieid range from', dfr['movieId'].min(), 'to', dfr['movieId'].max())
    print('Userid range from', dfr['userId'].min(), 'to', dfr['userId'].max())
    print('Rating number:', len(dfr))
    print('The unique values for ratings are : ', dfr['rating'].unique())
    print('Sparcity of the dataset is: ', (1 - len(dfr) / (len(dfr['movieId'].unique()) * len(dfr['userId'].unique()))) * 100, '%')

# Distribution of ratings in the dataset
if __name__=="__main__":
        rating = dfr['rating'].value_counts(ascending=True)
        fig = px.pie(values=rating, names=rating.index, title='Distribution of ratings in the dataset', width=800, height=350)
        fig.update_layout(
                autosize=False,
                title_x=0.5,
                width=800,
                height=800
        )
        fig.show()

# Movie count by genre
if __name__=="__main__":
    dfm['genres'] = dfm['genres'].apply(lambda x: x.split('|'))
    movies_df_exploded = dfm.explode('genres')
    px.histogram(movies_df_exploded, x='genres', height=400, title='Movie count by genre').update_xaxes(categoryorder="total descending")

