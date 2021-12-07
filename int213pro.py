#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[100]:


movies = pd.read_csv("movies.csv",usecols=['movieId','title']) 
movies.head()


# In[83]:


ratings = pd.read_csv("ratings.csv",usecols=['userId','movieId','rating']) 
ratings.head()


# In[103]:


ratings.shape


# In[129]:


movies_user = ratings.pivot(index='movieId',columns='userId',values='rating').fillna(0)
movies_user.head()


# In[147]:


import scipy.sparse 


# In[158]:


from scipy.sparse import csr_matrix


# In[161]:


mat_movies=csr_matrix(movies_user.values)


# In[169]:


import sklearn


# In[174]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=20)
model.fit(mat_movies)


# In[176]:


import fuzzywuzzy


# In[177]:


from fuzzywuzzy import process


# In[237]:


def recommender(movie_name, data,n):
    idx = process.extractOne (movie_name, movies['title'])[2]
    print('Movie Selected : ',movies ['title'][idx], 'Index: ',idx)
    print('Searching for recommondation.....')
    distance, indices = model.kneighbors (data[idx], n_neighbors=n) 
    for i in indices :
          print (movies['title'][i].where(i!=idx))


# In[262]:


recommender('Toy Story', mat_movies ,10)

