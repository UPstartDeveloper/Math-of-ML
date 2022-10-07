#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendations HW

# **Name: Syed Muhammad Zain Raza**

# **Collaboration Policy:** Homeworks will be done individually: each student must hand in their own answers. Use of partial or entire solutions obtained from others or online is strictly prohibited.

# **Late Policy:** Late submission have a penalty of 2\% for each passing hour. 

# **Submission format:** Successfully complete the Movie Lens recommender as described in this jupyter notebook. Submit a `.py` and an `.ipynb` file for this notebook. You can go to `File -> Download as ->` to download a .py version of the notebook. 
# 
# **Only submit one `.ipynb` file and one `.py` file.** The `.ipynb` file should have answers to all the questions. Do *not* zip any files for submission. 

# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# In[177]:


# Import all the required libraries
import numpy as np
import pandas as pd


# ## Reading the Data
# Now that we have downloaded the files from the link above and placed them in the same directory as this Jupyter Notebook, we can load each of the tables of data as a CSV into Pandas. Execute the following, provided code.

# In[178]:


# Read the dataset from the two files into ratings_data and movies_data
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, engine="python")


# In[179]:


print(ratings_data.shape)
ratings_data.head()


# In[180]:


column_list_movies = ["MovieID","Title","Genres"]
# for reasons I don't know - I passed an arg for "encoding" here
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, engine="python", encoding='unicode_escape')


# In[181]:


print(movies_data.shape)
movies_data.head()


# In[182]:


movies_data["Title"].nunique()  # just curious


# In[183]:


a = movies_data["MovieID"].unique() # just curious
a.sort()
a[:70]


# `ratings_data`, `movies_data`, `user_data` corresponds to the data loaded from `ratings.dat`, `movies.dat`, and `users.dat` in Pandas.

# In[184]:


column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, engine="python")


# In[185]:


print(user_data.shape)
user_data.head()


# In[186]:


user_data["UserID"].nunique() # just curious to see if there are duplicates


# ## Data analysis

# We now have all our data in Pandas - however, it's as three separate datasets! To make some more sense out of the data we have, we can use the Pandas `merge` function to combine our component data-frames. Run the following code:

# In[187]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)
print(data.shape)
data.head()


# Next, we can create a pivot table to match the ratings with a given movie title. Using `data.pivot_table`, we can aggregate (using the average/`mean` function) the reviews and find the average rating for each movie. We can save this pivot table into the `mean_ratings` variable. 

# In[188]:


mean_ratings=data.pivot_table('Ratings','Title',aggfunc='mean')
mean_ratings


# Now, we can take the `mean_ratings` and sort it by the value of the rating itself. Using this and the `head` function, we can display the top 15 movies by average rating.

# In[189]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by = 'Ratings',ascending = False).head(15)
top_15_mean_ratings


# Let's adjust our original `mean_ratings` function to account for the differences in gender between reviews. This will be similar to the same code as before, except now we will provide an additional `columns` parameter which will separate the average ratings for men and women, respectively.

# In[190]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
mean_ratings


# We can now sort the ratings as before, but instead of by `Rating`, but by the `F` and `M` gendered rating columns. Print the top rated movies by male and female reviews, respectively.

# In[191]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings.head(15))


# In[192]:


top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# Zain: the next cell adds an additional column, just to see the difference between the avg rating given by men and women for a given film?

# In[193]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# Let's try grouping the data-frame, instead, to see how different titles compare in terms of the number of ratings. Group by `Title` and then take the top 10 items by number of reviews. We can see here the most popularly-reviewed titles.

# In[194]:


ratings_by_title=data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# Similarly, we can filter our grouped data-frame to get all titles with a certain number of reviews. Filter the dataset to get all movie titles such that the number of reviews is >= 2500.

# In[195]:


ratings_by_title_over_2500 = ratings_by_title[ratings_by_title > 2500]


# In[196]:


# sanity check
if ratings_by_title_over_2500.min()  > 2500:
    print("It worked!")


# ## Question 1

# Create a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. Every element $[i,j]$ is a rating by user $i$ for movie $j$. Print the **shape** of the matrix produced.
# 
# **Notes:**
# - Do *not* use `pivot_table`.
# - A ratings matrix is *not* the same as `ratings_data` from above.
# - If you're stuck, you might want to look into the `np.ndarray` datatype and how to create one of the desired shape.
# - Every review lies between 1 and 5, and thus fits within a `uint8` datatype, which you can specify to numpy.

# In[197]:


from collections import OrderedDict


# In[198]:


# Create the matrix
unique_user_ids = data["UserID"].unique()
unique_movie_ids = movies_data["MovieID"].unique()

# init matrix
ratings = np.zeros((unique_user_ids.shape[0], unique_movie_ids.shape[0]), dtype=np.uint8)

# populate the matrix - 1 row (aka user) at a time
for index_user, uid in enumerate(unique_user_ids):
    user_ratings_per_movie = data[data["UserID"] == uid]
    user_ratings_per_movie_sorted = user_ratings_per_movie.sort_values("MovieID")
    # add 0s back in for shape compatibility
    user_ratings_per_movie_sparse = OrderedDict(zip(
        movies_data["MovieID"].unique(),
        np.zeros(unique_movie_ids.shape[0])
    ))  
    for _, rating_row in user_ratings_per_movie_sorted.iterrows():
        movie_id = rating_row[1]
        user_ratings_per_movie_sparse[movie_id] = rating_row[2]
    ratings[index_user][:] = list(user_ratings_per_movie_sparse.values())


# In[199]:


# Print the shape - expected shape is [6040, 3883]?
print(ratings.shape)


# ## Question 2

# Normalize the ratings matrix (created in **Question 1**) using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# - All of the `NaN` values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps. 
# - Your first step should be to get the average of every *column* of the ratings matrix (we want an average by title, not just by user!).
# - Second, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every row. It may be very close but not exactly zero because of the limited precision `float`s allow.
# - Lastly, divide this by the standard deviation of the *column*
# 
# - While creating the ratings matrix, you might not get any NaN values but if you look closely, you will see 0 values. This should be treated as NaN values as the original data does not have 0 rating. This should be replaced by the average.

# In[200]:


# 1: which axis to get the mean? ---> target shape is (1, 3883) --> axis 0
avg_rating_per_movie = ratings.mean(axis=0)
print(avg_rating_per_movie.shape)
np.isnan(avg_rating_per_movie).sum()


# In[201]:


#2: replacing NaN
ratings_no_nan = np.copy(ratings)
for movie_index in range(ratings.shape[1]):
    movie_ratings = ratings[:][movie_index]
    movie_ratings_no_nan = np.nan_to_num(movie_ratings, nan=avg_rating_per_movie[movie_index])
    ratings_no_nan[:][movie_index] = movie_ratings_no_nan


# Quick sanity check: are there any remaining `NaN` values?

# In[202]:


np.isnan(ratings_no_nan).sum()


# In[203]:


stddev = np.std(ratings_no_nan, axis=0).shape
(np.isnan(stddev)).sum()


# In[204]:


# 2: standardization
ratings_normalized = (ratings_no_nan - avg_rating_per_movie) / stddev
print(ratings_normalized.shape)
np.isnan(ratings_normalized).sum()


# ## Question 3

# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the $U$, $S$, and $V$ matrices you calculated.

# In[205]:


# Compute the SVD of the normalised matrix
U, S, V = np.linalg.svd(ratings_normalized)
S = np.diag(S)


# In[206]:


# Print the shapes
print(U.shape, S.shape, V.shape)


# ## Question 4

# Reconstruct four rank-k rating matrix $R_k$, where $R_k = U_kS_kV_k^T$ for k = [100, 1000, 2000, 3000]. Using each of $R_k$ make predictions for 3 users (select them from the dataset) for the movie with ID 1377 (Batman Returns).

# In[207]:


# which col did we put Batman Returns in? -->
batman_rating_index = list(user_ratings_per_movie_sparse.keys()).index(1377)

# construct each rating matrix at the desired k-value
for k in [100, 1_000, 2_000, 3_000]:
    # make 1 matrix
    rank = k
    rating_matrix_approx = U[:, :rank] @ S[:rank, :rank] @ V[:rank, :]
    # pick 3 users (I'll just choose the first three user IDs)
    print(f"==== SVD report, rank = {k} ====")
    for user_id in [1, 2, 3]:
        # lookup their rating for "Batman Returns" - i.e. the first 3 rows for the col
        print(f"Prediction for User {user_id} for 'Batman Returns': {rating_matrix_approx[user_id - 1][batman_rating_index]}.")


# ## Question 5

# ### Cosine Similarity
# Cosine similarity is a metric used to measure how similar two vectors are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. Cosine similarity is high if the angle between two vectors is 0, and the output value ranges within $cosine(x,y) \in [0,1]$. $0$ means there is no similarity (perpendicular), where $1$ (parallel) means that both the items are 100% similar.
# 
# $$ cosine(x,y) = \frac{x^T y}{||x|| ||y||}  $$

# **Based on the reconstruction rank-1000 rating matrix $R_{1000}$ and the cosine similarity,** sort the movies which are most similar. You will have a function `top_cosine_similarity` which sorts data by its similarity to a movie with ID `movie_id` and returns the top $n$ items, and a second function `print_similar_movies` which prints the titles of said similar movies. Return the top 5 movies for the movie with ID `1377` (*Batman Returns*):

# In[208]:


# Sort the movies based on cosine similarity
def top_cosine_similarity(data, movie_id, top_n=5):
    # grab the vector for the movie we want to find similar movies with
    all_movie_ids = list(user_ratings_per_movie_sparse.keys())
    movie_of_interest_id = all_movie_ids.index(movie_id)  # scalar
    # movie_of_interest_ratings = x = data[:][movie_of_interest_id]  # for some reason the shape of this array is (3883,1)?
    movie_of_interest_ratings = x = np.array([data[row][movie_of_interest_id] for row in range(data.shape[0])])  # (6080, 1)
    # Calculate the similarity to all the other vectors
    cosine_similarity = lambda y: (x.T @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    similarities = cosine_similarity(data)  # expected to have a shape of (3883,)
    # sort by top 5 greatest
    top_5_indices_desc_order = np.flip(np.argsort(similarities))[:5]
    return [all_movie_ids[index] for index in top_5_indices_desc_order]


def print_similar_movies(movie_data, movieID, top_indexes):
    print('Most Similar movies: ')
    sorted_similar_movie_ids = top_cosine_similarity(movie_data, movieID, top_indexes)
    # sorted_similar_movie_ids = list(range(1, 6)) # dummy data
    for index in range(top_indexes):
        movie_id = sorted_similar_movie_ids[index]
        movie_title = movies_data[movies_data["MovieID"] == movie_id]["Title"].values[0]
        print(f"{index + 1}. {movie_title}.")

# Print the top 5 movies for Batman Returns
movie_id = 1377
rank = 1000
data = U[:, :rank] @ S[:rank, :rank] @ V[:rank, :]
print_similar_movies(data, 1377, 5)

