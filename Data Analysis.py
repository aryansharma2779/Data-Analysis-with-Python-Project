
# coding: utf-8

# <font size=5><b>Data Analysis with Python</b></font>

# <font size=4><b>House Sales in King County,USA</b></font>

# <font size=3><b>
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.</b><br>
# 
# <b>id :</b>a notation for a house<br>
# 
# <b>date:</b> Date house was sold<br>
# 
# <b>price:</b> Price is prediction target<br>
# 
# <b>bedrooms:</b> Number of Bedrooms/House<br>
# 
# <b>bathrooms:</b> Number of bathrooms/bedrooms<br>
# 
# <b>sqft_living:</b> square footage of the home<br>
# 
# <b>sqft_lot: </b>square footage of the lot<br>
# 
# <b>floors :</b>Total floors (levels) in house<br>
# 
# <b>waterfront :</b>House which has a view to a waterfront<br>
# 
# <b>view: </b>Has been viewed<br>
# 
# <b>condition :</b>How good the condition is Overall<br>
# 
# <b>grade: </b>overall grade given to the housing unit, based on King County grading system<br>
# 
# <b>sqft_above :</b>square footage of house apart from basement<br>
# 
# <b>sqft_basement:</b> square footage of the basement<br>
# 
# <b>yr_built :</b>Built Year<br>
# 
# <b>yr_renovated :</b>Year when house was renovated<br>
# 
# <b>zipcode:</b>zip code<br>
# 
# <b>lat: </b>Latitude coordinate<br>
# 
# <b>long:</b> Longitude coordinate<br>
# 
# <b>sqft_living15 :</b>Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area<br>
# 
# <b>sqft_lot15 :</b>lotSize area in 2015(implies-- some renovations)<br>
# </font>

# <font size=5><b>Importing the Libraries</b></font>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# <font size=4><b>Getting Data From https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv
#     Converting to DataFrame</b></font>

# In[2]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# <font size=4><b>Observing Data</b></font>

# In[3]:


df.head()


# <font size=4><b>Checking Data Types</b></font>

# In[4]:


df.dtypes


# <font size=4><b>Observing diffrent Columns Data</b></font>

# In[5]:


df.describe()


# <font size=4><b>as we dont need colums such as "id","Unnamed" so we will Drop unwanted columns</b></font>

# In[6]:


df.drop(["id","Unnamed: 0"],axis=1,inplace=True)
df.describe()


# <font size=4><b>Checking Whether columns have null values of not</b></font>

# In[7]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# <font size=4><b>Replacing Null Values with means of the respective Columns</b></font>

# In[8]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[9]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[10]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# <font size=4><b>Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.</b></font>

# In[11]:


unique_floor=df["floors"].value_counts().to_frame()
unique_floor


# <font size=4><b>Plotting Box Plot</b></font>

# In[12]:


sns.boxplot(df["waterfront"],df["price"])


# <font size=4><b>Checking whether price and sqft_above</b></font>

# In[13]:


sns.regplot(df[["sqft_above"]],df["price"],data=df,ci=None)
plt.ylim(0,)


# <font size=4><b>Correlation Values of all Columns in ascending order</b></font>

# In[14]:



df.corr()['price'].sort_values()


# <font size=4><b>importing LinearRegression from sklearn</b></font>

# In[15]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# <font size=4><b>Training Linear Regression model between df["long"] and df["price"] <br>
#     Calculating R^2 value</b></font>

# In[16]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# <font size=4><b>Training Linear Regression with df["sqft_living"] and df["price"]<br>Calculating R^2</b></font>

# In[17]:


lm1=LinearRegression()
lm1.fit(df[["sqft_living"]],df["price"])
lm1.score(df[["sqft_living"]],df["price"])


# <font size=4><b>Checking R^2 Value for Multiple Linear Regression</b></font>

# In[18]:


features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]


# In[19]:



lm2=LinearRegression()
lm2.fit(features,df["price"])
lm2.score(features,df["price"])


# <font size=4><b>Importing method for Splitting data into Training and Testing dataset</b></font>

# In[20]:


from sklearn.model_selection import train_test_split
print("done")


# In[21]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# <font size=4><b>Performing Ridge Regression</b></font>

# In[22]:


from sklearn.linear_model import Ridge


# In[23]:



Rige=Ridge(alpha=.1)
Rige.fit(x_train,y_train)
Rige.score(x_test,y_test)


# <font size=4><b>Performing Polynomial  Regression of Degree 2</b></font>

# In[24]:



    

pk=PolynomialFeatures(degree=2)
x_train_pk=pk.fit_transform(x_train)
x_test_pk=pk.fit_transform(x_test)
rig=Ridge(alpha=.1)
rig.fit(x_train_pk,y_train)
rig.score(x_test_pk,y_test)




