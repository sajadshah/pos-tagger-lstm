
# coding: utf-8

# In[1]:


# Load our word2vec model
import gensim
w2v_model = gensim.models.word2vec.Word2Vec.load("/home/sajad/PycharmProjects/pos-tagger-lstm/exp/bijankhan_lbl_small/word2vec.model")
print("Model loaded")


# In[5]:


#Select 10000 words from our vocabulary
import numpy as np
count = 10000
word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
word_list = []
i = 0
for word in w2v_model.wv.vocab:
    word_vectors_matrix[i] = w2v_model[word]
    word_list.append(word)
    i = i+1
    if i == count:
        break
print("word_vectors_matrix shape is ", word_vectors_matrix.shape)


# In[6]:


#Compress the word vectors into 2D space
import sklearn.manifold
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
print("word_vectors_matrix_2d shape is ", word_vectors_matrix_2d.shape)


# In[7]:


import pandas as pd
points = pd.DataFrame(
    [
        (word, coords[0], coords[1]) 
        for word, coords in [
            (word, word_vectors_matrix_2d[word_list.index(word)])
            for word in word_list
        ]
    ],
    columns=["word", "x", "y"]
)
print("Points DataFrame built")


# In[14]:


points.head(10)


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("poster")


# In[11]:


points.plot.scatter("x", "y", s=10, figsize=(20, 12))


# In[34]:


from bidi.algorithm import get_display
import arabic_reshaper
def plot_region(x_bounds, y_bounds):
    
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1]) 
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():


        ax.text(point.x + 0.005, point.y + 0.005, get_display(arabic_reshaper.reshape(point.word)), fontsize=12)


# In[36]:


plot_region(x_bounds=(-20, 20), y_bounds=(-20, 20))


# In[46]:


[(w[0].decode('utf-8'), w[1]) for w in w2v_model.most_similar(".")]

