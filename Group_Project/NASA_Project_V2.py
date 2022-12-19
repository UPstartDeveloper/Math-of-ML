#!/usr/bin/env python
# coding: utf-8

# # CS 556 Final Group Project: Near-Earth Objects
# 
# ### *Jay Talekar, Jaydeep Maganbhai Dobariya, Syed Z. Raza*

# ## Imports 

# In[1]:


pip install -U scikit-learn  # because we'll be using DecisionBoundaryDisplay


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Get the Data
# 
# Let's use the version of the NASA dataset released on [GitHub](https://github.com/UPstartDeveloper/near-objects-dataset/releases/tag/v1) (so you don't need to worry about uploading anything to Google Drive). 

# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system(' pwd')
get_ipython().system(' ls ')


# Let's create a directory to hold the dataset (you can skip the next 3 cells if you've already ran it before):

# In[3]:


get_ipython().run_line_magic('mkdir', '/content/drive/MyDrive/NASA/')


# In[4]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/NASA/')


# In[5]:


get_ipython().system('wget https://github.com/UPstartDeveloper/near-objects-dataset/releases/download/v1/NASA.csv')


# ## Exploratory Data Analysis

# ### Example Rows from the Dataset

# In[6]:


df = pd.read_csv("./NASA.csv")
df.head()


# ### Missing Data?
# 
# Are there any missing values in our dataset? To answer this question, let's use Seaborn. In thee following visualization, we use a heatmap to highlight where the NaN values are in each column, if any:

# In[7]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Heatmap of NaNs in NASA Dataset")
plt.show()


# In[8]:


np.sum(df.isnull())


# **Takeaway:** woohoo! we have no NaNs. 
# 

# ### Redundant Features?
# 
# One of our priorities in creating machine learning models is to inherit a *low-variance, low-bias model*. Put differently, we want our model to be able to understand the full range of values that can influence the final label, without paying attention to features that don't really add useful information for making our classification.
# 
# OK having said this - do we have any columns in our dataset that only contain 1 unique value?

# In[9]:


sentry_dist = df['sentry_object'].value_counts()
sentry_dist.values


# (I have a hunch there is only 1 value in this column - let's visualize):

# In[10]:


plt.pie(sentry_dist.values, labels=sentry_dist.keys())
plt.title('Relative Frequency of Sentry Objects')
plt.legend()
plt.show()


# Let's also do this for the other categorical predictor variable - i.e., the `orbiting_body`. For the fun, I'll do so using another kind of visualization:

# In[11]:


orbit_dist = df['orbiting_body'].value_counts()

plt.hist(orbit_dist.values, label=orbit_dist.keys())
plt.title('Relative Frequency of Orbiting Bodies')
plt.legend()
plt.show()


# **Takeaway:** well then! It looks like both the `orbiting_body` and `sentry_object` columns are constant! So we can go ahead and drop them from the dataset, once we start modeling.
# 
# On a related note - the `id` column probably will not offer any relevant information for making classifications either. So we'll drop that as well:

# In[12]:


df_dropped = df.drop(labels=['id', 'orbiting_body', 'sentry_object'], axis=1)
df_dropped.head()


# ### Hazardous NEOs Over Time?
# 
# We observe that in the `name` column, the years of each near Earth object (NEO) is provided. Wouldn't it be interesting to see if the number of hazardous NEOs over time is trending up or down?
# 
# To do this - let's begin by engineering a new column that contains just the year of each record in the dataset:

# In[13]:


def extract_year(name: str) -> int:
    start_index = name.find("(") + 1
    year_chars = list(name)[start_index:start_index + 4]
    return int(''.join(year_chars))

  
def locate_names_with_no_year(names: str) -> str:
    years, nonyears = [], []
    for index, name in enumerate(names):
        try:
            years.append(extract_year(name))
        except ValueError:
            nonyears.append((index, name))
    return years, nonyears


# Let's try to confirm if we can actually find a year in all of these names:

# In[14]:


years, year_not_found = locate_names_with_no_year(df_dropped["name"])
year_not_found


# Interesting! So not every row has a year (and one that did have a year in 2019, was not picked up by our extraction function). 
# 
# Let's on the flip side, let's try to confirm if all the substrings we did catch in `years`, is in fact a valid year value:

# In[15]:


# if this runs without errors it means we probably don't have "false positive" years
def extract_year_with_nans(name: str) -> int:
    try:
        return extract_year(name)
    except ValueError:
        return np.nan


years_as_int = df_dropped["name"].apply(extract_year_with_nans)


# In[16]:


years_as_int[73482]


# In[17]:


# re-insert the one year that should be here, but got turned to NaN
years_as_int[73482] = 2019


# In[18]:


years_as_int.describe()


# So the good news is it looks like our years are already shuffled in the dataset for us. 
# But, we do have a few values that are too high to be an actual year - those I can turn to NaN.
# 
# And then we'll also see in total, how many records do we actually have that we were not able to find a year for:
# 

# In[19]:


def is_too_high(year):
    if year > 2022:
        return np.nan
    return year


# In[20]:


years_as_int = years_as_int.apply(is_too_high)


# In[21]:


np.isnan(years_as_int).sum()


# OK! So it looks like out of 90K+ records, there are only 10 for which we could not find a year for. Since this is so small, it should be ok for us to exclude them from the next visualization.
# 
# Now I'm curious to plot what the trend of hazardous NEOs looks like against time:

# In[22]:


years_and_hazards = pd.DataFrame(data={
    "Year": years_as_int,
    "Is_Hazard": df_dropped["hazardous"]
})
years_and_hazards.describe()


# In[23]:


aggregate_hazard_yrs = years_and_hazards.groupby("Year", sort=True).sum()


# In[24]:


plt.plot(aggregate_hazard_yrs.index,
         aggregate_hazard_yrs.values)
plt.title("Number of Hazardous NEOs Annually")
plt.ylabel('# Hazardous NEOs')
plt.xlabel('Year')
plt.show()


# **Takeaway:** interesting! Without knowing too much about the domain, our team is not sure what to make of this trend. The observation is that the number of hazardous NEOs seems to spike around the turn of the century. However, it does not explain why. To be on the conservative side, we presume that this spike is merely due to extraneous factors (e.g. perhaps in improved detection technology by NASA). It does not seem likely that the probability of a NEO being hazardous should suddenly spike because of the year it occurs.
# 
# Going forward, we will focus on trying to build a classifier that predicts just based on the actual physical properties of a given NEO, and leave the year out of the dataset.

# In[25]:


df_dropped2 = df_dropped.drop(labels=["name"], axis=1)
df_dropped2.head()


# In[26]:


df_dropped2.describe()


# ## Feature & Model Selection
# 
# The goals of this section are: 
# 
# 1. to determine which features we want to actually use for the model,
# 2. and on the flip side, which classifier type we want to use for the chosen features - a logistic regression, or support vector machine (potentially making use of the kernel trick?) 

# ### Data Preprocessing and Data Splitting
# 
# This will let us avoid "data snooping bias" for the rest of this project:

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[28]:


X = df_dropped2.drop("hazardous", axis=1)
y = df_dropped2["hazardous"]


# In[29]:


y = LabelEncoder().fit_transform(y.values)


# In[30]:


y = pd.DataFrame(data=y, columns=["hazardous"])  # verify we now have numbers
y.value_counts()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# ### Checking Correlations

# In[32]:


sns.heatmap(X_train.corr(), annot=True)
plt.title("Correlation Matrix of NASA Dataset")
plt.show()


# **Takeaway:** apart from `est_diameter_min` and `est_diameter_max`, there does not seem to be any strong multicolinearity between features in this dataset. 
# 
# So it would be prudent to drop either the `est_diameter_min` or `est_diameter_max` column, to help prevent overfitting.
# 
# This is just a hunch, but let's decide to drop the `est_diameter_max` before going forward - we presumably want our model to be very sensitive to the size of an NEO (after all, we don't want to miss any incoming hazards!); in this way, our model will learn to pick up on even small potential hazards.
# 
# Before going into next steps, let's standardize our predictor variables as well:

# In[33]:


X_train = X_train.drop("est_diameter_max", axis=1)
X_test = X_test.drop("est_diameter_max", axis=1)


# In[34]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ### Can We Use Logistic Regression? (TLDR - probably not a good move)
# 
# Based on this [blog post](https://www.statology.org/assumptions-of-logistic-regression/) on Statology by "Zach" (2020) [1], we know there are multiple assumptions we should check before using logistic regression. Let's check those now - if we cannot meet them all, then definitely we can still model using support vector machines in the project.
# 
# The good news, several of the assumptions in Zach's blog have already been verified here. Let's list them:
# 
# 1. "The Response Variable is Binary" - this is true, as we know we can only have hazardous/non-hazardous NEOs
# 2. "The Observations are Independent" - our dataset is not a time series dataset, so this is true
# 3. "There is No Multicollinearity Among Explanatory Variables" - based on the correlation heatmap above, we know this is true 
# 
# 
# 

# #### What About Assumption \#4? (Outlier Detection is Ambiguous)
# 
# As we know, logistic regression is sensitive to the presence of outliers. For this dataset, it is risky to try logistic regression, as we know of no clear way to distinguish outliers in multivariate distributions. Yes, we could attempt to try using the Mahalanobis Distance [[2]](https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3), but that involves computing the inverse of the covariance matrix - thus, it would likely consume all our memory and crash the notebook session (because our dataset is fairly large).
# 
# If we have time it might still be interesting to try logistic regression for this problem, because the size of the dataset might also help the model ignore any outliers altogether. But for now, it seems safer to try a support vector machine.

# ## Modeling (4 Features)

# ### 1: SVM (Lagrange Multipliers)

# #### Searching for the Best Kernel and Class Weight
# 
# Because we know this dataset is non-balanced, we will grid search for the right class weight. And we also don't know if we have linear separability or not, so for this first model let's try to also search for the best kernel to use:

# In[35]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[36]:


param_grid1 = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "class_weight": [None, "balanced"]  # this is less important to search for right now
} 


# In[37]:


# grid1 = GridSearchCV(SVC(),
#                      param_grid1,
#                      refit=True,
#                      cv=2,  # skipping cross validation for now
#                      verbose=2)
# grid1.fit(X_train_scaled, np.squeeze(y_train))


# ^Takes too long to fit, so Colab crashes before it can complete. Instead, we'll try a more efficient way of doing this via gradient descent:

# ### 2: Linear SVM (Gradient Descent)

# In[38]:


param_grid2 = {
    "class_weight": [None, "balanced"],
    "penalty": ["l2"],  # let's avoid l1, b/c we've already selected features 
    "random_state": [42],  # for reproducibility purposes
    "average": [True, False],
    "max_iter": [5000],
    "alpha": [.0001, .01, 10], # regularization param
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.001],  # we can tune this further, but for now I
                      # just want to see if using an 
                      # adaptive schedule vs. an optimal one has any value
                      # in the first place
    # these next two will help us not waste compute time
    "early_stopping":  [True],
    "validation_fraction": [0.1]
} 


# Note: this is just using a linear kernel for now

# In[39]:


grid2 = GridSearchCV(SGDClassifier(loss="hinge"),
                     param_grid2,
                     refit=True,
                     cv=5,  # skipping cross validation for now
                     verbose=2)
grid2.fit(X_train_scaled, np.squeeze(y_train))


# OK nice! We've tried a best of combinations - let's see what the best parameters turned out to be:

# In[40]:


grid2.best_params_


# Now, let's evaluate the best model so far:

# In[41]:


from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)


# In[42]:


ConfusionMatrixDisplay.from_estimator(grid2.best_estimator_,
                                      X_test_scaled, y_test)
plt.title("Linear SVM (SGD) Confusion Matrix")
plt.show()


# In[43]:


print(
    classification_report(y_test,
                          grid2.best_estimator_.predict(X_test_scaled))
)


# In[44]:


type(grid2.best_estimator_.class_weight)


# Good news: the test accuracy is high!
# 
# Bad news: the model has totally given up on being predicting the hazardous NEOs.
# 
# What can we do to remedy this, so that we improve the number of true positives predicted by our model?

# ### 3: Linear SVM (Balanced Class Weights, Gradient Descent)
# 
# Let's create a model that's nearly the same as before, but just with one small change - this time, we will weigh the positive (aka, the minority class) higher in our optimization:

# In[45]:


from sklearn.model_selection import cross_validate


# In[46]:


params_to_use = grid2.best_params_.copy()
params_to_use["class_weight"] = "balanced"
model3 = SGDClassifier(**params_to_use)
model3 = model3.fit(X_train_scaled, np.squeeze(y_train))


# In[47]:


ConfusionMatrixDisplay.from_estimator(model3,
                                      X_test_scaled,
                                      y_test)
plt.title("Linear SVM (SGD) Confusion Matrix, Balanced")
plt.show()


# In[48]:


print(
    classification_report(y_test,
                          model3.predict(X_test_scaled))
)


# **Takeaways:**
# 
# 1. *Good news*: the recall for our positive class has indeed improved! 
# 
# 2. *Bad News:* Our test accuracy has taken a hit :( 
#   
# 3. Even though this would be a much safer model to use in the real world, perhaps there is some way we can achieve high accuracy for both classes?

# ### 4: Kernel SVM (RBF, Balanced Class Weights, Gradient Descent)
# 
# The intuition here is that it's possible that our classes are not linearly separable. So, perhaps the key to building a highly accurate AND highly precise model is to use a non-linear kernel. Let's try that now:

# In[49]:


from sklearn.kernel_approximation import Nystroem


# In[50]:


# these args are totally arbitrary, just copying the docs for now:
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem

feature_map_nystroem1 = Nystroem(gamma=.2,
                                random_state=1,
                                n_components=300)


# In[51]:


best_params4 = {
    'alpha': 0.0001,
    'average': True,
    'class_weight': "balanced",
    'early_stopping': True,
    'eta0': 0.001,
    'learning_rate': 'optimal',
    'max_iter': 5000,
    'penalty': 'l2',
    'random_state': 42,
    'validation_fraction': 0.1
}


# In[52]:


X_train_rbf = feature_map_nystroem1.fit_transform(X_train_scaled)
X_test_rbf = feature_map_nystroem1.transform(X_test_scaled)


# In[53]:


model4 = SGDClassifier(**best_params4)
model4 = model4.fit(X_train_rbf, np.squeeze(y_train))


# In[54]:


ConfusionMatrixDisplay.from_estimator(model4,
                                      X_test_rbf,
                                      y_test)
plt.title("Kernel SVM (RBF, SGD) Confusion Matrix, Balanced")
plt.show()


# In[55]:


print(
    classification_report(y_test,
                          model4.predict(X_test_rbf))
)


# **Takeaways**:
# 
# 1. *Pros*: this model has a better recall on the positive class than our 3rd model
# 
# 2. *Cons*: our overall accuracy has still not really improved by that much.
# 
# Let's try one more time, just to see if tuning the feature map can help:

# In[56]:


feature_map_nystroem2 = Nystroem(gamma=.9,
                                random_state=1,
                                n_components=300)


# In[57]:


X_train_rbf2 = feature_map_nystroem2.fit_transform(X_train_scaled)
X_test_rbf2 = feature_map_nystroem2.transform(X_test_scaled)


# In[58]:


model5 = SGDClassifier(**best_params4)
model5 = model5.fit(X_train_rbf2, np.squeeze(y_train))


# In[59]:


ConfusionMatrixDisplay.from_estimator(model5,
                                      X_test_rbf2,
                                      y_test)
plt.title("Kernel SVM (RBF, SGD) Confusion Matrix, Balanced")
plt.show()


# In[60]:


print(
    classification_report(y_test,
                          model5.predict(X_test_rbf2))
)


# It does not help :(
# 
# Next - this is becoming more of an outlier detection problem. Perhaps we can see if a one-class support vector machine can help us achieve both a high accuracy, as well as high precision and recall for both classes?

# ### 5: One Class SVM
# 
# Like before:
# - we'll use gradient descent to optimize the model, due to the size of our dataset
# - we'll use an RBF kernel, due to the nature of our data probably being nonlinear (as we saw earlier, using the RBF kernel improved our recall)
# - the difference is, now we'll train the model only to recognize the non-hazardous examples - in this way, anything that is hazardous can potentially be immediately flagged as being out of the normal (said differently, as an "outlier")

# In[61]:


from sklearn.linear_model import SGDOneClassSVM


# In[62]:


# start by reloading the data, and separating both classes
data = df_dropped2.drop("est_diameter_max", axis=1)
# encode the class column, since I forgot to do before
data["hazardous"] = LabelEncoder().fit_transform(data["hazardous"])
train, test = train_test_split(data, test_size=.2, random_state=42) 


# In[63]:


train.head()  # sanity check


# We'll also have to measure the percentage of positive examples in our dataset again - this will be how we'll "weigh" this one class model:

# In[64]:


num_positive = train[train["hazardous"] == 1].shape[0]
outlier_proportion = num_positive / float(train.shape[0])


# Now, we're almost ready to train the one-class model. But I want to make sure we can use the kernel trick, so let's go ahead and transform this data.
# 
# We'll make a few different feature maps this time, so it'll help us in tuning the `gamma` parameter of our RBF:

# In[65]:


gammas = [0.00001, 0.001, 0.01, 0.05, 0.1, 1, 5, 10, 100]
gamma_kernel = dict()

for g in gammas:
    feature_map = Nystroem(gamma=g,
                           random_state=1,
                           n_components=300)
    # splitting claases in training data + processing
    train_normal, train_outlier = (
        train[train["hazardous"] == 0],
        train[train["hazardous"] == 1],
    )
    X_train, y_train = (
        train_normal.drop("hazardous", axis=1),
        train_normal["hazardous"],
    )
    X_test, y_test = (  # note: we include both classes in the test data
        test.drop("hazardous", axis=1),
        test["hazardous"],
    )
    scaler = StandardScaler()
    X_train_scaled_normal = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # apply the kernel
    X_train_rbf = feature_map.fit_transform(X_train_scaled_normal)
    X_test_rbf = feature_map.transform(X_test_scaled)
    gamma_kernel[g] = (X_train_rbf, X_test_rbf, y_train, y_test)
    


# Let's make sure the hyperparameters we set match what Scikit-learn expects:

# In[66]:


params_to_use5 = {
    'average': True,
    'eta0': 0.001,
    'learning_rate': 'optimal',
    'max_iter': 5000,
    'random_state': 42,
}


# Let's get training:

# In[67]:


models = dict()

for g, X in gamma_kernel.items():
    X_train_rbf, _, _, _ = X
    svm = SGDOneClassSVM(nu=outlier_proportion,
                         **params_to_use5)
    models[g] = svm.fit(X_train_rbf)
    


# And, let's get testing!

# In[68]:


def evaluate_one_class_svm(model, X_test, y_test, gamma=None):
    y_pred = model.predict(X_test)
    # relabel y_pred so -1 --> our positive class, and 1 --> our negatives
    y_pred_transformed = np.where(y_pred == -1, 1, 0)
    # visualize the confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_transformed)
    plt.title(f"One-Class SVM (RBF) Confusion Matrix, g = {gamma}")
    plt.show()
    print(classification_report(y_test, y_pred_transformed))


# In[69]:


results = list()

for g, X in gamma_kernel.items():
    _, X_test_rbf, _, y_test = X
    svm = models[g]
    evaluate_one_class_svm(svm, X_test_rbf, y_test, gamma=g)


# **Takeaways**:
# 
# - We know there are lots of graphs above - suffice to say, upon closer inspection we observe that the One-Class SVM has the highest number of both true positives and true negatives, and has its `gamma` parameter equal to `1`.
# 
# - **Restated: the best model has `gamma = 1`, its test accuracy is `0.85`, and the macro average of its precision, recall, and f1-score are all `0.59`**
# 
# - it doesn't seem likely we can tune the model any further than this and get noticeable improvement - as we can see from above, models that have gamma values both >1 and <1 begin to start favoring the majority class (of negatives) over the positive class
# 

# We'll need this `y_pred` for when we save predictions to a CSV later:

# In[71]:


gamma = 1
_, X_test, _, y_test = gamma_kernel[gamma]
svm = models[gamma]
y_pred_4_features = svm.predict(X_test)


# ## Dimensionality Reduction using PCA

# ### Visualizing, Standardizing and Splitting

# In[72]:


data.head() #sanity check


# #### Visualizing each feature wrt other features using Scatterplot

# In[73]:


g = sns.PairGrid(data, hue="hazardous")
g.map_offdiag(sns.scatterplot)
g.map_diag(sns.histplot, multiple='stack')
g.add_legend()


# In[74]:


fig, axes = plt.subplots(1, 4, figsize = (20, 5))
fig.suptitle("Features vs Target Variable")
sns.boxenplot(ax=axes[0], x='hazardous',y='est_diameter_min',data=data)
sns.boxenplot(ax=axes[1], x='hazardous',y='relative_velocity',data=data)
sns.boxplot(ax=axes[2], x='hazardous',y='miss_distance',data=data)
sns.boxplot(ax=axes[3], x='hazardous',y='absolute_magnitude',data=data)
plt.show()


# Splitting the data into Features and Target Varible

# In[75]:


X = data.drop(labels='hazardous', axis=1)
y = data['hazardous']


# Splitting data into further into Train and Test Dataset ans standardizing the them

# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[77]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Finding 2 Principal Components with PCA

# 

# In[78]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)


# ## Modeling (2 Features)

# ### 1: Linear SVM (Gradient Descent & w/o Balanced Class Weights) 

# In[79]:


param_grid2_reduced = {
    "class_weight": [None],
    "penalty": ["l2"],  # let's avoid l1, b/c we've already selected features 
    "random_state": [42],  # for reproducibility purposes
    "average": [True, False],
    "max_iter": [5000],
    "alpha": [.0001, .01, 10], # regularization param
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.001],  
    "early_stopping":  [True],
    "validation_fraction": [0.1],
} 


# In[80]:


grid2_reduced = GridSearchCV(SGDClassifier(loss="hinge"),
                     param_grid2_reduced,
                     refit=True,
                     cv=5,  # skipping cross validation for now
                     verbose=2)
grid2_reduced.fit(X_train_reduced, np.squeeze(y_train))


# In[81]:


grid2_reduced.best_params_


# In[82]:


ConfusionMatrixDisplay.from_estimator(
    grid2_reduced.best_estimator_,
    X_test_reduced, y_test
)
plt.title("Linear SVM (SGD) Confusion Matrix (2 Features)")
plt.show()


# In[83]:


print(classification_report(y_test, 
                            grid2_reduced.best_estimator_.predict(X_test_reduced)))


# ### 2: Linear SVM (Gradient Descent & Balanced Class Weights) 

# In[84]:


params_grid3_reduced = grid2_reduced.best_params_.copy()
params_grid3_reduced['class_weight'] = 'balanced'
model3_reduced = SGDClassifier(**params_grid3_reduced)
model3_reduced.fit(X_train_reduced, y_train)


# In[85]:


ConfusionMatrixDisplay.from_estimator(model3_reduced,
                                      X_test_reduced, y_test)
plt.title("Linear SVM (SGD) Confusion Matrix, Balanced (2 Features)")
plt.show()


# In[86]:


print(classification_report(y_test,
                            model3_reduced.predict(X_test_reduced)))


# ### 3: Kernel SVM (RBF, Balanced Class Weights, Gradient Descent

# In[87]:


params_grid4_reduced = params_grid3_reduced.copy()


# In[88]:


feature_map_nystroem_reduced = Nystroem(gamma=.9,
                                random_state=1,
                                n_components=300)


# In[89]:


X_train_reduced_rbf = feature_map_nystroem_reduced.fit_transform(X_train_reduced)
X_test_reduced_rbf = feature_map_nystroem_reduced.transform(X_test_reduced)


# In[90]:


model4_reduced = SGDClassifier(**params_grid4_reduced)
model4_reduced = model4_reduced.fit(X_train_reduced_rbf, y_train)


# In[91]:


ConfusionMatrixDisplay.from_estimator(model4_reduced, X_test_reduced_rbf, y_test)
plt.title("Kernel SVM (RBF, SGD) Confusion Matrix, Balanced (2 Features)")
plt.show()


# In[92]:


print(classification_report(y_test,
                            model4_reduced.predict(X_test_reduced_rbf)))


# ### 4: One Class SVM

# In[93]:


num_positive = X_train_reduced[y_train == 1].shape[0]
outlier_proportion = num_positive / float(X_train_reduced.shape[0])


# In[94]:


gammas = [0.00001, 0.001, 0.01, 0.05, 0.1, 1, 5, 10, 100]
gamma_kernel_reduced = dict()

for g in gammas:
    feature_map = Nystroem(gamma=g,
                           random_state=1,
                           n_components=300)
    # splitting claases in training data + processing
    X_train, y_train = (
        X_train_reduced[y_train == 0], y_train
    )
    X_test, y_test = (  # note: we include both classes in the test data
        X_test_reduced, y_test
    )
    scaler = StandardScaler()
    X_train_scaled_normal = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # apply the kernel
    X_train_rbf = feature_map.fit_transform(X_train_scaled_normal)
    X_test_rbf = feature_map.transform(X_test_scaled)
    gamma_kernel_reduced[g] = (X_train_rbf, X_test_rbf, y_train, y_test)
    


# In[95]:


params4 = {
    'average': True,
    'eta0': 0.001,
    'learning_rate': 'optimal',
    'max_iter': 5000,
    'random_state': 42,
}


# In[96]:


models_reduced = dict()

for g, X in gamma_kernel_reduced.items():
    X_train_rbf, _, _, _ = X
    svm = SGDOneClassSVM(nu=outlier_proportion,
                         **params4)
    models_reduced[g] = svm.fit(X_train_rbf)
    


# In[97]:


for g, X in gamma_kernel_reduced.items():
    _, X_test_rbf, _, y_test = X
    svm = models_reduced[g]
    evaluate_one_class_svm(svm, X_test_rbf, y_test, gamma=g)


# ## Performance optimization using other models
# 
# After the implementation of parametric models such as SVM, we have decided to try non-parametric models too, such as **Decision Tree and Random Forest**, which make no assumptions about the distribution of data and are not influenced by outliers and multicollinearity to some fair extent.
# 
# 
# 

# In[106]:


def evaluation(y_test,y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    print('Classification report:\n',classification_report(y_test,y_pred))
    plt.show()


# ### 1: Decision Tree

# In[107]:


from sklearn.tree import DecisionTreeClassifier

x=data.drop(columns='hazardous')
y=data.hazardous 


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y, random_state = 42)

DT = DecisionTreeClassifier()

DT_model = DT.fit(x_train,y_train)
y_pred = DT_model.predict(x_test)

evaluation(y_test,y_pred)


# ### 2: Random Forest

# In[108]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

RF_model = RF.fit(x_train,y_train)
y_pred = RF_model.predict(x_test)

evaluation(y_test,y_pred)


# ## Conclusion

# ### Visualization of PCA Model
# 
# For this plot we'll use the PCA that used one-class SVC, since it appeared to perform the best (out of the other classifiers trained on principal components) on recognizing samples from both classes in the test set.

# For our visualization, we will be using the  `DecisionBoundaryDisplay` class. 

# In[98]:


from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap


# Setup code:

# In[99]:


gamma = 1
_, X_test, _, y_test = gamma_kernel_reduced[gamma]
svm = models_reduced[gamma]

pipe = make_pipeline(feature_map_nystroem_reduced, svm)

red_green_colormap = ListedColormap(["green", "red"])


# Plotting function:

# In[100]:


def plot_svm_2D(title, clf, X_test, y_test, gamma):
    '''Shows the plot of our SVM in color.'''
    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(figsize=(6, 6))
    # plot the samples and decision boundary
    X0, X1 = X_test[:, 0], X_test[:, 1]

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        cmap=red_green_colormap,
        alpha=0.8,
        ax=ax,
        xlabel="PC 1",
        ylabel="PC 2",
        plot_method="contour",
        grid_resolution=100
    )
    scatter = ax.scatter(X0, X1,
               c=y_test, 
               cmap=red_green_colormap,
               edgecolor="black",
               s=20, edgecolors='k')
    #add legend
    lines, _ = scatter.legend_elements()
    labels = ["Non-hazardous", "Hazardous"]
    new_legend_elems = (lines, labels)
    plt.legend(*new_legend_elems)

    # final presentation pieces
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()


# In[101]:


plot_svm_2D(f"One Class SGD-SVC in 2D, gamma = {gamma}",
            pipe, X_test_reduced, y_test, gamma)


# This visualization corroborates our quantitative results:
# 
# - there are far more non-hazardous points than hazardous ones
# - we can also see that the decision boundary (colored in red) encircles an area of what appears to be alot of green points - this is in line with our confusion matrix, which indeed showed **a high number of false positives**
# - also - the fact that the decision boundary is roughly a circle is indicative of how we used the RBF kernel, rather than just a linear one
# 
# Note: please forgive us for highlighting the decision boundary in red - I know the instructions said to use black but the red color stands out better.

# ### Before and After PCA - *which model is most effective, and why?*
# 
# Ultimately, in comparing the One Class SVM model trained on 4 features vs. the one trained on 2 principal components (PCs), we observe that, with `gamma = 1`, both models achieve an accuracy of 85%. However, the model trained on 4 features scores higher on `precision, recall, and f1-score` with a value of `0.59`; on the other hand, the models trained on PCs achieves a slightly lower score of `0.57`. All of the afforementioned metrics are with respect to the test set. This difference in perfomance suggests the model trained on 4 features is more effective.
# 
# Why is this so? We believe there are three main reasons why PCA fails to improve our model on the Near-Earth Objects (NEOs) dataset:
# 
# 1. **Non-linearity**: PCA is best suited to datasets where the two classes are linearly separable. This was not the case for the NEOs dataset. Therefore, even though we reduced dimensionality, that itself didn't remove the problem of trying to fit a support vector machine to the dataset.
# 2. **Non-orthogonal structure**: PCA works by finding an axis along which to project the samples of our dataset onto, such that we preserve maximal information about the spread of our values. This projection needs to happen orthogonally, so that the PCs will be linearly separable; however for our dataset, we can see that our PCs kind of form one large "clump" in 2D. This indicates that PCA didn't help much for this dataset, since we don't improve the linear separability of our data. And as discussed above, this is bad news for training SVMs. 
# 
# **Other conclusions** from this project is regarding our use of the RBF kernel. Rather than finding a clear winner between the RBF and linear kernels, we observe a tradeoff: while our SVM trained on a linear kernel had a slightly higher test F1-score, the RBF kernel increased our models recall value on the "hazardous" class, in exchange for slight F1-score decrease (< 2%).
# 

# ### Next Steps:
# 
# Our group would be excited to pursue further work in this problem domain. Here are a few ideas we have already considered:
# 
# 1. **Further feature engineering or data collection** - while we used the RBF kernel to increase the dimensionality of our dataset, in the future it would be interesting to see if we could add features for each NEO that related to real-world phenomenon, e.g. looking at elemental properties of the object. In this way, we have a hunch we could potentially find even better dimensions along which to separate out "hazardous" vs. "non-hazardous" samples.
# 2. **Comparing other kinds of models** - for this project, we were narrowly focused on using support vector machines (and variants thereof, e.g. one-class svm) to classify this dataset. In the future, it would be curious to? experiment with utilizing non-parametric classifiers for this problem such as decision trees, or perhaps even ensemble techniques. 

# ### Final Prediction CSV

# In[111]:


models_providing_preds = [
    # the col name,      the model,     # the X_test to use
    ('SVM (4 Features)', models[gamma], gamma_kernel[gamma][1]),
    ('SVM (2 PCs)',      models[gamma], gamma_kernel[gamma][1]),
    ('Decision Tree',    DT_model,      x_test),
    ('Random Forest',    RF_model,      x_test),
]


# In[112]:


data_for_df = {"y_true": y_test}

for index_name, estimator, X_test in models_providing_preds:
    y_pred = estimator.predict(X_test)
    data_for_df[index_name] = y_pred

# finally, create the CSV and save
df = pd.DataFrame(data_for_df)
df.to_csv("./Ground_Truth_and_Model_Predictions.csv")


# In[ ]:




