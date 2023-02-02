print("Loading data...")
from dataprocessing import *
print("Finished loading data.")
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **************************
'''NeuralMF model for personalised recommender'''
# **************************

# Define MLP model -> Everytime I read MLP I'll think of 'My Little Pony' -> Cartoon ;)
class MLP(nn.Module):
    def __init__(self, n_userIds, n_movieIds, dims=[32,64,32,16]):
        super().__init__()
        # Set embedding dimension
        self.User_Embedding = nn.Embedding(num_embeddings = n_userIds, embedding_dim = dims[0] // 2)
        self.Movie_Embedding = nn.Embedding(num_embeddings = n_movieIds, embedding_dim = dims[0] // 2)
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
        )

    def forward(self, u, m):
        # Get user and movie embedding
        u_embed = self.User_Embedding(u)
        m_embed = self.User_Embedding(m)
        # Concatenate two embeddings together before put them into the MLP layers
        x = torch.cat([u_embed, m_embed], dim=-1)
        # Feed it to the pony ;) -> MLP
        out = self.linear_relu_stack(x)
        return out

# Define GMF model 
class GMF(nn.Module):
    def __init__(self, n_userIds, n_movieIds, dims=10):
        super().__init__()
        self.User_Embedding = nn.Embedding(num_embeddings = n_userIds, embedding_dim = dims)
        self.Movie_Embedding = nn.Embedding(num_embeddings = n_movieIds, embedding_dim = dims)

    def forward(self, u, m):
        # Get user and movie embedding
        u_embed = self.User_Embedding(u)
        m_embed = self.User_Embedding(m)
        
        # Calculate the element-wise product of the input embeddings
        out = torch.mul(u_embed, m_embed)
        return out

# Define NeuralMF model
class NeuralMF(nn.Module):
    def __init__(self, n_userIds, n_movieIds, mlp_dims=[32,64,32,16], gmf_dims=10):
        super().__init__()
        self.mlp = MLP(n_userIds, n_movieIds, mlp_dims)
        self.gmf = GMF(n_userIds, n_movieIds, gmf_dims)
        
        self.output_layer = nn.Sequential(
            nn.Linear(gmf_dims + mlp_dims[-1], 1),
            nn.Sigmoid(),
        )
        
    def forward(self, u, m):
        # Get the output of mlp and gmf
        mlp = self.mlp(u,m)
        gmf = self.gmf(u,m)
        # Concatenate two mlp and gmf together
        x = torch.cat([mlp, gmf], dim=-1)
        # Feed it to NeuMF layer
        out = self.output_layer(x)
        return out
    
model = NeuralMF(n_userIds, n_movieIds, mlp_dims=[32,64,32,16], gmf_dims=10).to(device)

# Pre-trained weights for the model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# testUser = torch.tensor([[13]]).to(device)
# testMovie = torch.tensor([[0]]).to(device)
# print(model(testUser, testMovie))

'''Recommendation for the system'''
# For each user, store a list of all the movies they haven't seen
# Used the model for the unseen list for the user, predict and recommend the top 30 film the user will like
def rec_personalised(userid, df):
    n_movieIds = df['movieId'].unique()
    all_movies = set(range(n_movieIds))
    # the movies the current user has seen
    seen = df.loc[df['userId'] == userid, 'movieId'].values
    # the movies the current user has not seen 
    unseen = list(all_movies - set(seen))
    # pass the userid and the list of movie unseen by the user to the model
    pred_movie = torch.tensor(unseen).to(device)
    pred_user = torch.tensor(userid).to(device)
    # Get the prediction from the model
    pred = model(pred_user, pred_movie)[:,0] # remove 2nd dimension
    # Sort the prediction by index, get the top ones
    idx = torch.argsort(pred, descending=True)[:TOP]
    # Get the recommendation
    recommendation = pred_movie[idx.cpu()].cpu().tolist()
    
    return recommendation