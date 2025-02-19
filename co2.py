#!/usr/bin/env python
# coding: utf-8

# # Business Objective:
# ## The fundamental goal here is to model the CO2 emissions as a function of several car engine features.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


# ## EDA

# In[2]:


df = pd.read_csv("C:\\Users\\prana\\Downloads\\co2_emissions (1).csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum() #checking for missing values


# In[7]:


df.duplicated().sum() # checking for duplicated values


# In[8]:


df.drop_duplicates(inplace=True) # removing duplicates


# In[9]:


df.shape


# In[10]:


df.describe() # statistical summary


# In[11]:


df.nunique()


# In[12]:


#seperating numerical columns and categorical columns
cat = df.columns[df.dtypes == 'O']
num = df.columns[df.dtypes != 'O']


# ### -Visualization

# In[13]:


# box plot for identifying outliers
for col in num:
    plt.figure(figsize=(10,5))
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.grid()


# In[14]:


# histogram
for col in num:
    plt.figure(figsize=(10,5))
    sns.histplot(df[col] ,kde=True)
    plt.title(col)
    plt.grid()


# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(df[['fuel_consumption_city','fuel_consumption_hwy']], bins=30, kde=True)
plt.title('Fuel Consumption Distribution')
plt.show()


# In[16]:


#count plot for catagorical values
for col in cat:
    plt.figure(figsize=(10,4))
    sns.countplot(x=df[col])
    plt.title(col)
    plt.xticks(rotation=90)
    plt.grid()


# In[17]:


# pair plot
sns.pairplot(df)
plt.show()


# In[18]:


#correlation heat map


# In[19]:


# Bivariate analysis: CO2 emissions vs other featuresfor col in num:
if col != 'co2_emissions':
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[col], y=df['co2_emissions'], alpha=0.7, color='purple')
    plt.title(f'CO2 Emissions vs {col}')
    plt.xlabel(col)
    plt.ylabel('CO2 Emissions')
    plt.show()


# In[20]:




# ### Insights

# #### * Larger engines with more cylinders are less fuel-efficient and emit more CO2.
# #### * Fuel consumption metrics in liters/100 km are strongly related to each other, as are their inverse measures in mpg.
# #### * Fuel efficiency (mpg) is inversely correlated with both fuel consumption and emissions, emphasizing the environmental benefit of fuel-efficient vehicles.

# ### -Data Cleaning

# In[21]:


# We have to remove Natural Gas from our data set because it has only one entry
df_natural=df[df['fuel_type']=="Natural Gas"]
natural=df_natural.index
df_natural
for i in natural:
    df.drop(i, axis = 0,inplace = True)


# ## Normalize

# ### -Standard Scaler

# In[22]:


# apply StandardScaler to numerical columns
scaler = StandardScaler()
df[num] = scaler.fit_transform(df[num])


# ### -Label Encoder

# In[23]:


# apply LabelEncoder to catagorical columns
encoder = LabelEncoder()
df[cat]=df[cat].apply(encoder.fit_transform)

corr_matrix=df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix ,cmap='Spectral', annot = True ,square = True ,vmin=-1 , vmax=1 ,center=0)
plt.show()

print("correlation between matrix")
print( corr_matrix)

# ## Model Building & Evaluation

# In[24]:


# seperating features and target columns
X = df.drop(['co2_emissions'], axis= 1) # features
y = df["co2_emissions"] # target


# In[25]:


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


# function for evaluation
def evaluate(y_test, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Results:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}\n")


# In[27]:


# function for visualization
def visualize(y_test, y_pred, model_name="Model"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.show()


# ### -Linear regression

# In[28]:


lr=LinearRegression() # building model
lr.fit(X_train, y_train) # train model
y_pred_lr = lr.predict(X_test) # prediction


# In[29]:


lr_evaluation=evaluate(y_test, y_pred_lr,'Linear regression') # evaluation
lr_evaluation


# In[30]:


visualize(y_test, y_pred_lr,'Linear regression') # visualization


# ### -Random Forest

# In[31]:


rt=RandomForestRegressor(n_estimators=100, random_state=42) # building model
rt.fit(X_train, y_train) # train model
y_pred_rt = rt.predict(X_test) # prediction


# In[32]:


rt_evaluation=evaluate(y_test, y_pred_rt,'Random Forest') # evaluation
rt_evaluation


# In[33]:


visualize(y_test, y_pred_rt,'Random Forest') # visualization


# ### -SVR

# In[34]:


svr=SVR() # building model
svr.fit(X_train, y_train) # train model
y_pred_svr = svr.predict(X_test) # prediction


# In[35]:


svr_evaluation=evaluate(y_test, y_pred_svr,'SVR') # evaluation
svr_evaluation


# In[36]:


visualize(y_test, y_pred_svr,'SVR') # visualization


# ### -KNN

# In[37]:


knn=KNeighborsRegressor() # building model
knn.fit(X_train, y_train) # train model
y_pred_knn = knn.predict(X_test) # prediction


# In[38]:


knn_evaluation=evaluate(y_test, y_pred_knn,'KNN') # evaluation
knn_evaluation


# In[39]:


visualize(y_test, y_pred_knn,'KNN') # visualization


# ### -XGBoost

# In[40]:


xgb=XGBRegressor(objective='reg:squarederror', random_state=42) # building model
xgb.fit(X_train, y_train) # train model
y_pred_xgb = xgb.predict(X_test) # prediction


# In[41]:


xgb_evaluation=evaluate(y_test, y_pred_xgb,'XGBoost') # evaluation
xgb_evaluation


# In[42]:


visualize(y_test, y_pred_xgb,'XGBoost') # visualization


# ### -LightGBM

# In[43]:


lgbm=LGBMRegressor(random_state=42) # building model
lgbm.fit(X_train, y_train) # train model
y_pred_lgbm = lgbm.predict(X_test) # prediction


# In[44]:


lgbm_evaluation=evaluate(y_test, y_pred_lgbm,'LightGBM') # evaluation
lgbm_evaluation


# In[45]:


visualize(y_test, y_pred_lgbm,'LightGBM') # visualization


# ## Summary of Model Performance

# #### Best Models: XGBoost and Random Forest performed the best with lowest errors (RMSE < 0.055) and highest R² (~0.997), making them the most accurate.
# #### LightGBM: Also performed well (R² = 0.996), slightly behind XGBoost and Random Forest.
# #### Linear Regression & KNN: Decent but less effective (R² ≈ 0.88-0.91) with higher errors.
# #### SVR: Performed poorly (R² = 0.075), indicating it is not suitable for this task.

# ## Conclusion

# #### XGBoost is the best choice, followed closely by Random Forest and LightGBM, while SVR should be avoided.


pickle.dump(xgb , open("model.pkl", "wb"))

scaler_y = StandardScaler()
df['co2_emissions'] = scaler_y.fit_transform(df[['co2_emissions']])
with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)