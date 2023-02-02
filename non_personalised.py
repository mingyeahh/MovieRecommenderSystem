from dataprocessing import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TOP = 30
# Good rate threshold
GOOD = 7


############## The basic idea of the non-personalised recommender system  ##############
############## 1. the average ratings across users                        ##############
############## 2. watched time by users                                   ##############
############## 3. the number of genres the movie includes                 ##############
############## -> all of these are generic features                       ##############


''' The system for non-personalised recommender '''

def nps(df, top):
    # ------ Build feature data --------
    # 1. Get the averaage rating  for each film
    average_rating = df.groupby('movieId')['rating'].mean()
    df['averageRating'] = df['movieId'].map(average_rating)

    # 2. Get the number of rating for each film
    counts = df['movieId'].value_counts()
    df['rateTime'] = df['movieId'].map(counts)

    # 3. Get the number of genres for each film
    df['genresCount'] = df['genres'].str.count('\\|') + 1
    
    # Visualisation of features that have the same values -> reason why muilti-layers of sorting is needed
    # ar = 5855 - len(df[['averageRating']].value_counts())
    # print(f'{ar} films that have the same average rating.\n')
    # rt = 5855 - len(df[['averageRating', 'rateTime']].value_counts())
    # print(f'{rt} films that have the same average rating and the same number of rate time.\n')
    # gc = 5855 - len(df[['averageRating', 'rateTime', 'genresCount']].value_counts())
    # print(f'{gc} films that have the same average rating, the same number of rate time and the same number of genres.\n')
    
    # ------- Dataset cleaning -------
    # Drop columns that will affect further duplicate removing
    df = df.drop(['genres', 'rating', 'userId'], axis=1)

    # Drop duplicates in the dataset
    df = df.drop_duplicates()

    # Sort the dataset by the averageRating, ratetime and number of genres in order.
    df = df.sort_values(['averageRating', 'rateTime', 'genresCount'], ascending=[False, False, False])
    
    # Get the top films from the sorted list for recommendation
    r = df.head(top)
    return r


if __name__=="__main__":

    '''System Evaluation'''
    # Build evaluation dataframe
    # The Idea is to apply different metrics for evaluation. Metrics used: MSE, RMSE, Recall30 and Precision
    dfn = pd.merge(dfr, dfm, on='movieId')
    # Building train and testing set for further evaluation
    train_users, test_users = train_test_split(np.arange(n_userIds), test_size=0.3, random_state=3)
    train = dfn[dfn['userId'].isin(train_users)].copy()
    test = dfn[dfn['userId'].isin(test_users)].copy()

    # Get the list of top recommendations
    r = nps(train, TOP)

    movie_list = r['movieId'].values

    # Get rows that has the same movie id as the recommended 30 films
    e_df = test.loc[test['movieId'].isin(movie_list)]

    e_df = e_df.drop(['title', 'genres'], axis = 1)
    
    average_rating = train.groupby('movieId')['rating'].mean()
    # Make a new column for predicted rating
    e_df['pred'] = e_df['movieId'].map(average_rating)

    # Evaluation in the metric of mse and rmse
    mse = mean_squared_error(e_df['rating'].values, e_df['pred'].values)
    rmse = np.sqrt(mse)

    # Evaluation in the metric of precision and recall30
    all_rec = len(e_df)
    tp = len(e_df[(e_df['rating'] >= GOOD) & (e_df['pred'] >= GOOD)])
    precision =  tp / all_rec

    gt_good = len(test[(test['rating'] >= GOOD)])
    recall = tp / gt_good

    print('-- Evaluation for non-psersonalised recommender system --')
    print(f'Recommendation number is {TOP}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'percision: {precision}')
    print(f'recall: {recall}')