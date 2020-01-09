#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


CSV_PATH = "../input/rotten-tomatoes-movies-and-critics-datasets/rotten_tomatoes_movies.csv"
df = pd.read_csv(CSV_PATH)

print("Head (below): ")
df.head()


# In[3]:


# A little pre-processing.
# Make genre comma-separated -> non. comma separated
print("Creating individual genre columns [Animation, Drama, Horror, ...] and saving to data frame")
genre_cols = df.genre.str.get_dummies(sep=', ')
# print(genre_cols)
for col in genre_cols:
    df[col + '_genre'] = genre_cols[col].values

# Make ratings label-encoded
print("Replacing ratings [PG, G, R, ... etc] with numbers 0 - ...")
unique = df['rating'].unique()
print(unique)
label_encoder = LabelEncoder()
df.rating = label_encoder.fit_transform(df.rating)
unique = df['rating'].unique()
print(unique)
# print(df.rating)

# Then drop the old genre/cast/direc/writers columns
# df.drop(['genre', 'cast', 'directors', 'writers'], axis=1)
    
print("Columns provided: ")
print(df.columns)

print("Head (below): ")
df.head()


# In[4]:


df = df.dropna(axis=0)
# print(df) reveals no columns were dropped. Imputation not needed.
# unique = df['in_theaters_date'].unique()
# print(sorted(unique))

# y = df['tomatometer_status']

y = df['tomatometer_rating']
X = df[['rating', 'runtime_in_minutes', 'Action & Adventure_genre', 'Animation_genre', 'Anime & Manga_genre',
       'Art House & International_genre', 'Classics_genre', 'Comedy_genre',
       'Cult Movies_genre', 'Documentary_genre', 'Drama_genre',
       'Faith & Spirituality_genre', 'Gay & Lesbian_genre', 'Horror_genre',
       'Kids & Family_genre', 'Musical & Performing Arts_genre',
       'Mystery & Suspense_genre', 'Romance_genre',
       'Science Fiction & Fantasy_genre', 'Special Interest_genre',
       'Sports & Fitness_genre', 'Television_genre', 'Western_genre', 'studio_name']]

# Split our data
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# print(X_train.shape)
# print(" vs ")
# print(X_valid.shape)

# Determine columns that are categorical in nature
obj = (X.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical: ")
print(object_cols)

# one-hot encode rating, directors, writers, studio_name 
OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))

OH_cols.index = X.index

numerical_X = X.drop(object_cols, axis = 1)

OH_X = pd.concat([numerical_X, OH_cols], axis=1)

print("Complete shape of feature dataframe: ")
print(OH_X.shape)
#y_train.head()
# y_valid.head()

# Split our data
X_train, X_valid, y_train, y_valid = train_test_split(OH_X, y, random_state=0)

print("Verify that the columns are same size: ")
print(X_train.shape)
print(" vs ")
print(X_valid.shape)


# In[5]:


def mean_absolute_percentage_error(y_valid, y_prediction): 
    y_valid = np.array(y_valid)
    y_prediction = np.array(y_prediction)
    return np.mean(np.abs((y_valid - y_prediction) / y_valid)) * 100


# In[6]:


model = None;
predictions = None;

def train_and_validate(DX_train, DX_valid, Dy_train, Dy_valid) :
    global model
    global predictions 
    
    model = RandomForestRegressor(random_state = 1)
    model.fit(DX_train, Dy_train)
    
    predictions = model.predict(DX_valid)
    print(predictions)
    
    mae = mean_absolute_error(Dy_valid, predictions)
    # print("MAE = " + str(mae))
    
    return mae

def train_and_validate(mln, DX_train, DX_valid, Dy_train, Dy_valid) :
    global model
    global predictions 
    
    model = RandomForestRegressor(max_leaf_nodes=mln, random_state = 1)
    model.fit(DX_train, Dy_train)
    
    predictions = model.predict(DX_valid)
    print(predictions)
    
    print(" ... vs ...")
    print(Dy_valid)
    
    mae = mean_absolute_error(Dy_valid, predictions)
    # print("MAE = " + str(mae))
    
    return mae

def xg_train_and_validate(DX_train, DX_valid, Dy_train, Dy_valid) :
    global model
    global predictions 
    
    model = XGBRegressor(n_estimators=350, learning_rate=0.20, n_jobs=4, random_state=1)
    model.fit(DX_train, Dy_train, early_stopping_rounds=5, 
              eval_set=[(DX_valid, Dy_valid)], verbose=False)
    
    predictions = model.predict(DX_valid)
    print(predictions)
    
    print(" ... vs ...")
    print(Dy_valid)
    
    mae = mean_absolute_error(Dy_valid, predictions)
    # print("MAE = " + str(mae))
    print("MAPE = " + str(mean_absolute_percentage_error(Dy_valid, predictions)))
    
    return mae


# In[7]:


# This reveals ~400 max leaf nodes for the RandomForest model provides MAE of ~19:

for max_leaf_nodes in [4, 40, 400, 1600, 64000]:
    mae = train_and_validate(max_leaf_nodes, X_train, X_valid, y_train, y_valid)
    print("Max leaf nodes: %d  \t Mean Absolute Error:  %f" %(max_leaf_nodes, mae))


# In[8]:


# Predict with self.
print("MAE [self] = %d" %(xg_train_and_validate(X_train, X_train, y_train, y_train)) )

# Predict with validation data.
print("MAE [validation] = %d" %(xg_train_and_validate(X_train, X_valid, y_train, y_valid)) )


# In[9]:


print(predictions) # compare with y_valid
valid = y_valid.to_numpy() #indices, as you see above, are not neat (y_valid[737], y_valid[10978], ...). convert it to numpy array.    # .index.values
corresponding_rating = X_valid.rating.to_numpy() # removes indices.

print(predictions.shape)
print(valid.shape)

print(corresponding_rating)

hist_X = np.zeros(6) # => keep sum [G, NC17, NR, PG, PG-13 and PG-13), R and R)]
count_X = np.zeros(6) # => keep count

err = np.zeros( (predictions.shape[0]) )

for i in range(predictions.shape[0]):
    err[i] = abs(predictions[i] - valid[i])
    # print("This: " + str(corresponding_rating[i]) + " with error " + str(err[i]))
    if(corresponding_rating[i] == 0): #G
        hist_X[0] = (err[i] + hist_X[0])
        count_X[0] += 1
    elif (corresponding_rating[i] == 1): #NC-17
        hist_X[1] = (err[i] + hist_X[1])
        count_X[1] += 1
    elif (corresponding_rating[i] == 2): #NR
        hist_X[2] = (err[i] + hist_X[2])
        count_X[2] += 1
    elif (corresponding_rating[i] == 3): #PG
        hist_X[3] = (err[i] + hist_X[3])
        count_X[3] += 1
    elif (corresponding_rating[i] == 4 or corresponding_rating[i] == 5): #PG-13 || PG-13)
        hist_X[4] = (err[i] + hist_X[4])
        count_X[4] += 1
    elif (corresponding_rating[i] == 6 or corresponding_rating[i] == 7): #R || R)
        hist_X[5] = (err[i] + hist_X[5])
        count_X[5] += 1
        
        
for j in range(6):
    hist_X[j] = (hist_X[j] / float(count_X[j]))

print ("\nMean error compared with admission ratings...\n\t[G, NC17, NR, PG, PG-13, R] <=> " + str(hist_X))


# In[10]:


plt.figure(figsize=(6,7))
plt.ylim(0, 100)
plt.title("Mean absolute error for predicting Rotten Tomatoes\ncritic scores in comparison with film ratings")
plt.ylabel("Mean absolute error")
plt.xlabel("Film rating")
s_plot = sns.barplot(x=np.array(['G', 'NC17', 'NR', 'PG', 'PG-13', 'R']), y=hist_X)

