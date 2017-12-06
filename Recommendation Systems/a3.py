# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:56:50 2017

@author: Swapnil
"""

# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    
    tokens = []
    for val in movies["genres"]:
        tokens.append(tokenize_string(val))
    movies["tokens"] = tokens
    
    return movies
    
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    df = defaultdict(int)

    dict_movie = defaultdict()
    vocabulary = set()
    
    movie_no = movies.shape
    no_movies = movie_no[0]
    movies["features"] = ""
    
    for index, val in movies.iterrows():
        dict_movie[val["movieId"]] = (len(set(val["tokens"])),Counter(val["tokens"]),index)
       
        for token in val["tokens"]:
            df[token]= df[token]+1
            vocabulary |= {token}
        
    vocab = defaultdict()
    v = 0
    for term in sorted(vocabulary):
       # print(term)
        vocab[term] = v
        v= v + 1
        
    for movie in sorted(dict_movie):
        datastore = []
        colm =[]
        row = dict_movie[movie]
        maxval= dict_movie[movie][1].most_common(1)[0][1]
        
        for val in row[1]:
            #print (val)
            tf_id = row[1][val]
            df_i= df[val]
            div =(tf_id/maxval)
            logma = math.log((no_movies / df_i),10)
            tf_idf = div * logma
            datastore.append(tf_idf)
            colm.append(vocab[val])
        
        rowvals = len(colm)*[0]
        out = csr_matrix((datastore,(rowvals,colm)),shape=(1,len(vocab)))
        index = row[2]
        movies.set_value(index=index, col="features",value=out)
        
    return (movies,vocab)


    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    
    numerator = np.dot(a.toarray(),b.toarray().T)
    denominator= np.linalg.norm(a.toarray())*np.linalg.norm(b.toarray())
    
    div = numerator[0][0] / denominator
    
    return div 
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    
    rtc = ratings_test.copy(deep=True)
    
    mean= defaultdict(list)

    dict_user= defaultdict(list)
    
    for index, val in ratings_train.iterrows():
       # print (index)
        dict_user[val["userId"]].append((val["movieId"],val["rating"]))
        mean[val["userId"]].append(val["rating"])
    user=1.0
        
    while user in mean:
        #print(user)
        mean[user] = sum(mean[user])/ len(mean[user]) #mean ratings for every userID
        user= user +1
    for index, val in ratings_test.iterrows():
        rated_movie = dict_user[val["userId"]]
        denominator = 0.0
        numerator=0.0
        average_rating=0.0
        pos=0
        tmov = movies.loc[movies["movieId"] == val["movieId"],"features"] 
        temp =tmov.iloc[0]
        
        a = temp
        
        for m in rated_movie:
            #print (m)
            mov=movies.loc[movies["movieId"] == m[0],"features"]
            movie_row = mov.iloc[0]
            b= movie_row
            simab= cosine_sim(a,b)
            
            if(simab ==1 or simab>0):
                denominator = denominator + simab
                numerator = numerator + simab * m[1]
                pos = pos+1
                
        if(pos ==0):
            average_rating = mean[val["userId"]]
            
        elif(pos >0):
            average_rating = numerator/denominator
            
        #Printing average:
            
        
        rtc.set_value(index=index,col="rating",value=average_rating)
        
    numpyarr = rtc["rating"].values
    
    return numpyarr
    
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
