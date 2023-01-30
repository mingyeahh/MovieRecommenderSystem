print('Loading...\n')
from dataprocessing import *

# Get the movie id from the movie input by the user
def getMovieid(movieStr):
    while True:
        # Check whether the dataset contains the input string as a substring
        if (dfm['title'].str.contains(movieStr)).any():
            selection = dfm[dfm['title'].str.contains(movieStr)]
            print('There is a list of movies from the dataset based on your input.')
            print('Please choose one of them and \033[4menter the movie ID\033[0m:\n')
            print(f"{'ID':>4} | Title")
            print("-----+"+'-'*20)
            for i, row in selection.iterrows():
                print(f"{row['movieId']:>4} | {row['title']}")
                print('  ')
            movieid = input("\nEnter the movieid: ")
            while True:
                # Check whether the input movie id is valid -> is an integer and is on the given list
                if movieid.isnumeric() and int(movieid) in selection['movieId'].values:
                    return int(movieid)
                else:
                    movieid = input('Please enter a valid movieId on the given list: ')

        else:
            print('Sorry, the input movie is not in our database :(. ')
            print('Please check whether you have typed in the correct movie name or try another movie?')

            movieStr = input("\nEnter the film name: ")
            
# Check whether the input user id is valid            
def checkUserid(userid):
    while True:
        if userid.isnumeric() and int(userid) in dfr['userId'].values:
            return int(userid)
        else:
            userid = input('Please enter a valid userId: ')

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
        print('⭐ The system applies a pre-trained neural collaborative filtering model for movie recommendation,')
        print('which means it gives prediction by comparing your past preference(movie ratings) with other users who are similar to you and recommend to you some movies these users rank high :D! Meow~\n')
        print('⭐ Of course! If you don\'t want us to use your data, you can use the non-personalised system!')
        print('⭐ You can either tell us what type of film you want use to recommend or tell us the name of the past film you\'ve watched and let us recommend similar films to you.')
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
print('To use the system, you need to select a recommendation type first. Meow~')
print('- For a personalised one, you need to login so we could use your past ratings to predict what you might like among the films you haven\'t watched before!')
print('- For a non-personalised one, you don\'t need to log in, the recommendation will only be based on the input film, and find the similar films for you.\n')
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
        print('Please enter a film you like:')
        film = input("\nEnter the film name: ")
        m = getMovieid(film)
        print('Movie id is:',m)
        break

    else:
        print('Please enter a correct input.')

