from non_personalised import nps
print('Loading...\n')

# Get the movie id from the movie input by the user
# def getMovieid(movieStr):
#     while True:
#         # Check whether the dataset contains the input string as a substring
#         if (dfm['title'].str.contains(movieStr)).any():
#             selection = dfm[dfm['title'].str.contains(movieStr)]
#             print('There is a list of movies from the dataset based on your input.')
#             print('Please choose one of them and \033[4menter the movie ID\033[0m:\n')
#             print(f"{'ID':>4} | Title")
#             print("-----+"+'-'*20)
#             for i, row in selection.iterrows():
#                 print(f"{row['movieId']:>4} | {row['title']}")
#                 print('  ')
#             movieid = input("\nEnter the movieid: ")
#             while True:
#                 # Check whether the input movie id is valid -> is an integer and is on the given list
#                 if movieid.isnumeric() and int(movieid) in selection['movieId'].values:
#                     return int(movieid)
#                 else:
#                     movieid = input('Please enter a valid movieId on the given list: ')

#         else:
#             print('Sorry, the input movie is not in our database :(. ')
#             print('Please check whether you have typed in the correct movie name or try another movie?')

#             movieStr = input("\nEnter the film name: ")
            
# Check whether the input user id is valid (is numeric and is in the database)            
def checkUserid(userid):
    while True:
        if userid.isnumeric() and int(userid) in dfr['userId'].values:
            return int(userid)
        else:
            userid = input('Please enter a valid userId: ')

# User getting recommendation
def show_rec(df):
    rl = df['title'].values
    for i in range(len(rl)):
        print(f'[{i+1}]:  {rl[i]}')

# Intro to users about the system
print('~~ Hello there! Welcome to use this movie recommender system! Meow~~')
print('My name is Kiki🐱, a cat who loves watching movies! Do you want to know more about the system?\n')
print('(A) Yes, please.')
print('(B) Skip the introduction.')
login_action = input("\nEnter: ").upper()
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('-----------------------------------------------------------------------------------------')


while True:
    if login_action == 'A':
        print('🐱: The system offered two type of movie recommendation: a personalised one and a non-personalised one.')
        print('🐱: The personalised system applies a pre-trained neural collaborative filtering model for movie recommendation,')
        print('which means it gives prediction by comparing your past preference(movie ratings) with other users who are similar to you and recommend to you some movies these users rank high :D! Meow~ Thus, you will need to log in with your userid so we can store your user preference.\n')
        print('🐱: Of course! If you don\'t want your data to be used, you can use the non-personalised system, which will give recommendation according to the average ranking, number of watches and number of genre of movies in our database. Top 30 movies will be recommended according to the conditions mentioned before.')
        print('🐱: Hope you like the meowie recommender system! Meow!')
        print('\n')
        input("Press Enter to continue :")
        login_action = 'B'

    elif login_action == 'B':
        break


    else:
        print('Please enter a correct input.')
        login_action = input("\nEnter: ").upper()
        
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('-----------------------------------------------------------------------------------------')
print('🐱: To use the system, you need to select a recommendation type first. Meow~')
print('- For a personalised one, you need to login so we could use your past ratings to predict what you might like among the films you haven\'t watched before!')
print('- For a non-personalised one, you don\'t need to log in, the recommendations are selected by the system based on the database.\n')
print('Please choose the recommendation type:')
print('(A) Personalised recommendation. Log-in required.')
print('(B) Non-personalised recommendation. Log-in not required.')
print('  ')
print('Please choose the recommendation type:')
r_type = input("\nEnter: ").upper()
while True:
    if r_type == 'A':
        print('')
        print('Please enter your user ID:')
        print('  ')
        userid = input("\nEnter: ")
        u = checkUserid(userid)
        print('User id is:',u)
        break


    elif r_type == 'B':
        # Building dataset for non-personalised system
        dfn = pd.merge(dfr, dfm, on='movieId')
        r = nps(dfn, TOP)
        show_rec(df)
        break
        
        

    else:
        print('Please enter a correct input.')

