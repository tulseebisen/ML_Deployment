#!/usr/bin/env python
# coding: utf-8

# # Appliances energy prediction

# ### Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
from datetime import datetime
import time
import dateutil
import pickle


# ### Loading the datasets

# In[2]:


df = pd.read_csv("energydata_complete.csv",nrows=10)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# #### From above table we can see that there are no null values present in dataset

# In[7]:


df.describe()


# In[8]:


df.hist(figsize=(20, 20))


# In[9]:


# This function take a dataframe
# as a parameter and returning list
# of column names whose contents 
# are duplicates.
def getDuplicateColumns(df):
  
    # Create an empty set
    duplicateColumnNames = set()
      
    # Iterate through all the columns 
    # of dataframe
    for x in range(df.shape[1]):
          
        # Take column at xth index.
        col = df.iloc[:, x]
          
        # Iterate through all the columns in
        # DataFrame from (x + 1)th index to
        # last index
        for y in range(x + 1, df.shape[1]):
              
            # Take column at yth index.
            otherCol = df.iloc[:, y]
              
            # Check if two columns at x & y
            # index are equal or not,
            # if equal then adding 
            # to the set
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
                  
    # Return list of unique column names 
    # whose contents are duplicates.
    return list(duplicateColumnNames)
  
# Driver code
if __name__ == "__main__" :
    # Get list of duplicate columns
    duplicateColNames = getDuplicateColumns(df)
  

    print('Duplicate Columns are :')
        
    # Iterate through duplicate
    # column names
    for column in duplicateColNames :
       print('Column Name : ', column)


# #### rv1 and rv2 are having same values therefore we are dropping any 1 column  

# In[10]:


df=df.drop(["rv2"], axis = 1)


# ## Cheking for the Corelation among features

# In[11]:


fig, ax = plt.subplots(figsize=(20,20)) 
corr = df.corr()
sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)


# In[12]:


# Convert correlation matrix to 1-D Series and sort
corr2 = df.corr(method='pearson')
sorted_mat = corr2["Appliances"].sort_values()
  
print(sorted_mat)


# #### 
# It  can  be  seen  from  the  figure  that  all  temperature  characteristics  from  T1~T9  are positively  correlated  with  energy  consumption.  
# 
# For  indoor  temperature,  the  correlation  coefficient between T9 and T3, T5  and  T7  is greater than 0.9,  which  is  highly positively correlated with  them.   
# 
# For  outdoor  temperatures,  T6  has  a  correlation with TO of 0.97 and is highly positively correlated. Analysis  of  the  two temperature characteristics  of  T9  and T6, the information  they  provide  can  be provided  by  other  temperature  data,  
# 
#  It can be said that if there is no correlation,The correlation coefficient between two variables is near to 0,  and  the  correlation  is  extremely  low,  so  it  has  little effect.  Visibility  and    random  variables  can be removed from the data sets

# In[13]:


df=df.drop(["Visibility","rv1","RH_5","T9","Tdewpoint","RH_4"], axis = 1)


# In[14]:


df


# In[15]:


sns.countplot(df['Appliances'])


# #### From above graph we can see that the graph is positive skewed

# ## Spliting the date

# In[16]:


df["exact_date"]=df['date'].str.split(' ').str[0]

df["hours"]=(df['date'].str.split(':').str[0].str.split(" ").str[1]).astype(str).astype(int)
df["seconds"]=((df['date'].str.split(':').str[1])).astype(str).astype(int).mul(60)

df["days"]=(df['date'].str.split(' ').str[0])
df["days"]=(df['days'].apply(dateutil.parser.parse, dayfirst=True))
df["days_num"]=(df['days'].dt.dayofweek).astype(str).astype(int)
df["days"]=(df['days'].dt.day_name())


# In[17]:


df.groupby('exact_date')['Appliances'].plot()


# In[18]:


df.groupby('hours')['Appliances'].mean().plot(figsize=(10,8))
plt.xlabel('Hour')
plt.ylabel('Appliances consumption in Wh')
                                                           
plt.title('Mean Energy Consumption per Hour of a Day')


# In[19]:


df.head(3)


# In[20]:


import seaborn as sns
sns.boxplot(x=df['Appliances'])


# ### Above plot shows that there are many outliers, these are outliers as there are not included in the box of other observation i.e no where near the quartiles.

# In[21]:


X=df[['Appliances']]
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
print("Outlier threshold of Appliances ",IQR)

dfOutlier=X.describe()
outlierSet=set()
for column in dfOutlier.columns:
    Q1 = dfOutlier[column]['25%']
    Q3 = dfOutlier[column]['75%']
    IQR = Q3 - Q1
    outlierDf= ( ((X[column] < (Q1 - 1.5 * IQR)) |(X[column] > (Q3 + 1.5 * IQR))) )
    outlierSet.update(set(outlierDf[outlierDf==True].index))
        

df.drop(outlierSet, inplace=True, axis=0)


# ### Draph after removing Outlier

# In[22]:


sns.countplot(df['Appliances'])


# ### Droping some features as they are less impactful on target feature

# In[23]:


df=df.drop(["date","lights","exact_date","days","seconds","days_num"], axis = 1)


# In[24]:


df.shape


# In[25]:


df.head(3)


# In[26]:


df.columns


# ## Split dataset into Train and Test

# In[27]:


# selecting label and features
X = df.drop('Appliances',axis=1)
y= df['Appliances']  


# In[28]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2)


# In[29]:


from sklearn.preprocessing import StandardScaler

# Normalizing of X matrices for each model to mean = 0 and standard deviation = 1

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 


# # Linear Regressor

# In[30]:


from sklearn import linear_model

lin_model = linear_model.LinearRegression()
lin_model.fit(X_train,y_train)


# In[31]:


y_pred_lr = lin_model.predict(X_test)


# In[32]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lr))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_lr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))
print('r2_score:', metrics.r2_score(y_test, y_pred_lr)) 


# In[33]:


errors = abs(y_pred_lr - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# ## Random Forest Regressor

# In[34]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100,random_state=1)            
rf_model.fit(X_train, y_train)


# In[35]:


y_pred_rf = rf_model.predict(X_test)


# In[36]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print('r2_score:', metrics.r2_score(y_test, y_pred_rf)) 


# In[37]:


errors = abs(y_pred_rf - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# # pickle file

# In[38]:


# Saving model to disk
pickle.dump(rf_model, open('energy_model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('energy_model.pkl','rb'))
#print(model.predict([[2, 9, 6,57,43,32,54,62,,,,,,]]))

