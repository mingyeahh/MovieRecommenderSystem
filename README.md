# Meowie Recommender System
This project aims at providing two different types of movie recommender system - personalised and non-personalised, and comparing the differences between them. To achieve better performance, the personalised system is a collaborative filtering system that applies a deep learning technique (NCF)  with generalised matrix factorisation (GMF) and multi-layer perceptron (MLP) model architecture components. The non-personalised system is a system to output the highest ranked movies based on the database, conditioning on the average rating, watch time and diversity of the genres. Detailed information is presented in the [Recommender_System.pdf](Recommender_System.pdf) report.

## Getting Started
Here is how you could utilise the code.

### Prerequisites
*  Python3 (pip included)

### Installation
```sh
pip install -r requirements.txt
```

## Usage
* Downloud Dataset
Download MovieLens 10M10K dataset from [here](https://grouplens.org/datasets/movielens/10m/), unzip it and add the dataset folder to the project directory.

* Run
```
python3 user.py
```

You will be asked whether you wish to see an introduction, and then which type of recommendation (personalised or not) you would like. Select these options by entering `A` or `B` as prompted. If you choose personalised recommendations, you will be prompted for your user ID - these range between 0 and 43434, inclusive.
