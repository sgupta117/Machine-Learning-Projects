#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import pandas as pd
import numpy as np


# In[20]:


# Loading the dataset
df = pd.read_csv('ipl.csv')


# ## **Exploring the dataset**

# In[21]:


df.columns


# In[22]:


df.shape


# In[23]:


df.dtypes


# In[24]:


df.head()


# ## **Data Cleaning**
# Points covered under this section:<br/>
# *• Removing unwanted columns*<br/>
# *• Keeping only consistent teams*<br/>
# *• Removing the first 5 overs data in every match*<br/>
# *• Converting the column 'date' from string into datetime object*<br/>

# In[25]:


df.columns


# In[26]:


# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']

print('Before removing unwanted columns: {}'.format(df.shape))
df.drop(labels=columns_to_remove, axis=1, inplace=True)
print('After removing unwanted columns: {}'.format(df.shape))


# In[27]:


df.columns


# In[28]:


df.head()


# In[29]:


df.index


# In[30]:


df['bat_team'].unique()


# In[31]:


consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']


# In[32]:


# Keeping only consistent teams
print('Before removing inconsistent teams: {}'.format(df.shape))
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(df.shape))


# In[33]:


df['bat_team'].unique()


# In[34]:


# Removing the first 5 overs data in every match
print('Before removing first 5 overs data: {}'.format(df.shape))
df = df[df['overs']>=5.0]
print('After removing first 5 overs data: {}'.format(df.shape))


# In[35]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))


# In[36]:


# Selecting correlated features using Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Get correlation of all the features of the dataset
corr_matrix = df.corr()
top_corr_features = corr_matrix.index

# Plotting the heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(data=df[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# ## **Data Preprocessing**
# *• Handling categorical features*<br/>
# *• Splitting dataset into train and test set on the basis of date*<br/>

# In[37]:


# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
encoded_df.columns


# In[38]:


encoded_df.head()


# In[39]:


# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[41]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))


# ## **Model Building**
# I will experiment with 5 different algorithms, they are as follows:<br/>
# *• Linear Regression*<br/>
# *• Decision Tree Regression*<br/>
# *• Random Forest Regression*<br/>
# 
# ----- Boosting Algorithm -----<br/>
# *• Adaptive Boosting (AdaBoost) Algorithm*<br/>

# ### *Linear Regression*

# In[42]:


# Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,y_train)


# In[43]:


# Predicting results
y_pred_lr = linear_regressor.predict(X_test)


# In[44]:


# Linear Regression - Model Evaluation
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
print("---- Linear Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_lr))))


# ### *Decision Tree*

# In[45]:


# Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor
decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train,y_train)


# In[46]:


# Predicting results
y_pred_dt = decision_regressor.predict(X_test)


# In[47]:


# Decision Tree Regression - Model Evaluation
print("---- Decision Tree Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_dt)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_dt)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_dt))))


# ### *Random Forest*

# In[48]:


# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train,y_train)


# In[49]:


# Predicting results
y_pred_rf = random_regressor.predict(X_test)


# In[50]:


# Random Forest Regression - Model Evaluation
print("---- Random Forest Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_rf)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_rf)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_rf))))


# *Note: Since Linear Regression model performs best as compared to other two, we use this model and boost it's performance using AdaBoost Algorithm*

# ### *AdaBoost Algorithm*

# In[51]:


# AdaBoost Model using Linear Regression as the base learner
from sklearn.ensemble import AdaBoostRegressor
adb_regressor = AdaBoostRegressor(base_estimator=linear_regressor, n_estimators=100)
adb_regressor.fit(X_train, y_train)


# In[52]:


# Predicting results
y_pred_adb = adb_regressor.predict(X_test)


# In[53]:


# AdaBoost Regression - Model Evaluation
print("---- AdaBoost Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_adb)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_adb)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_adb))))


# *Note: Using AdaBoost did not reduce the error to a significant level. Hence, we will you simple linear regression model for prediction*

# ## **Predictions**
# • Model *trained on* the data from **IPL Seasons 1 to 9** ie: (2008 to 2016)<br/>
# • Model *tested on* data from **IPL Season 10** ie: (2017)<br/>
# • Model *predicts on* data from **IPL Seasons 11 to 12** ie: (2018 to 2019)

# In[35]:


def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
  temp_array = list()

  # Batting Team
  if batting_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Daredevils':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Daredevils':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
  temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

  # Converting into numpy array
  temp_array = np.array([temp_array])

  # Prediction
  return int(linear_regressor.predict(temp_array)[0])


# ### **Prediction 1**
# • Date: 16th April 2018<br/>
# • IPL : Season 11<br/>
# • Match number: 13<br/>
# • Teams: Kolkata Knight Riders vs. Delhi Daredevils<br/>
# • First Innings final score: 200/9
# 

# In[36]:


final_score = predict_score(batting_team='Kolkata Knight Riders', bowling_team='Delhi Daredevils', overs=9.2, runs=79, wickets=2, runs_in_prev_5=60, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 2**
# • Date: 7th May 2018<br/>
# • IPL : Season 11<br/>
# • Match number: 39<br/>
# • Teams: Sunrisers Hyderabad vs. Royal Challengers Bangalore<br/>
# • First Innings final score: 146/10
# 

# In[37]:


final_score = predict_score(batting_team='Sunrisers Hyderabad', bowling_team='Royal Challengers Bangalore', overs=10.5, runs=67, wickets=3, runs_in_prev_5=29, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 3**
# • Date: 17th May 2018<br/>
# • IPL : Season 11<br/>
# • Match number: 50<br/>
# • Teams: Mumbai Indians vs. Kings XI Punjab<br/>
# • First Innings final score: 186/8<br/>
# 

# In[38]:


final_score = predict_score(batting_team='Mumbai Indians', bowling_team='Kings XI Punjab', overs=14.1, runs=136, wickets=4, runs_in_prev_5=50, wickets_in_prev_5=0)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 4**
# • Date: 30th March 2019<br/>
# • IPL : Season 12<br/>
# • Match number: 9<br/>
# • Teams: Mumbai Indians vs. Kings XI Punjab<br/>
# • First Innings final score: 176/7
# 

# In[39]:


final_score = predict_score(batting_team='Mumbai Indians', bowling_team='Kings XI Punjab', overs=12.3, runs=113, wickets=2, runs_in_prev_5=55, wickets_in_prev_5=0)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 5**
# • Date: 11th April 2019<br/>
# • IPL : Season 12<br/>
# • Match number: 25<br/>
# • Teams: Rajasthan Royals vs. Chennai Super Kings<br/>
# • First Innings final score: 151/7
# 

# In[40]:


final_score = predict_score(batting_team='Rajasthan Royals', bowling_team='Chennai Super Kings', overs=13.3, runs=92, wickets=5, runs_in_prev_5=27, wickets_in_prev_5=2)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 6**
# • Date: 14th April 2019<br/>
# • IPL : Season 12<br/>
# • Match number: 30<br/>
# • Teams: Sunrisers Hyderabad vs. Delhi Daredevils<br/>
# • First Innings final score: 155/7
# 

# In[41]:


final_score = predict_score(batting_team='Delhi Daredevils', bowling_team='Sunrisers Hyderabad', overs=11.5, runs=98, wickets=3, runs_in_prev_5=41, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# ### **Prediction 7**
# • Date: 10th May 2019<br/>
# • IPL : Season 12<br/>
# • Match number: 59 (Eliminator)<br/>
# • Teams: Delhi Daredevils vs. Chennai Super Kings<br/>
# • First Innings final score: 147/9
# 

# In[42]:


final_score = predict_score(batting_team='Delhi Daredevils', bowling_team='Chennai Super Kings', overs=10.2, runs=68, wickets=3, runs_in_prev_5=29, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# In[55]:


# Create pickle file for the classifier
import pickle
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(linear_regressor, open(filename, 'wb'))


# *Note: In IPL, it is very difficult to predict the actual score because in a moment of time the game can completely turn upside down!*
# 
