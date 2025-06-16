#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("SupplyChainDataset.csv")
df.head(6)


# In[3]:


df.info()


# In[4]:


df_cust=df[['Customer City','Customer Country','Customer Email','Customer Fname','Customer Lname','Customer Id','Customer Password','Customer Segment','Customer State','Customer Street','Customer Zipcode']]


# In[5]:


print(df[df['Customer Zipcode'].isnull()])


# In[6]:


df_cust=df_cust.drop([35704,46440,82511])


# In[7]:


df=df.drop("Product Description", axis=1)


# In[8]:


df=df.drop([35704,46440,82511])


# In[9]:


df=df.drop('Order Zipcode',axis=1)


# In[10]:


df


# In[11]:


df_cust


# In[12]:


df_order=df[['Order City', 'Order Country','Order Customer Id','order date (DateOrders)','Order Id','Order Item Cardprod Id','Order Item Discount','Order Item Id','Order Item Product Price','Order Item Profit Ratio','Order Item Quantity','Sales','Order Item Total','Order Profit Per Order','Order Region','Order State','Order Status']]


# In[13]:


df_order


# In[20]:


df_order.describe()


# In[28]:


maxthresh=df['Sales'].quantile(0.95)
minthresh=df['Sales'].quantile(0.05)
maxthresh,minthresh


# In[30]:


df=df[(df['Sales']>minthresh)&(df['Sales']<maxthresh)]
df.describe()


# In[44]:


diff =df['Days for shipping (real)']-df['Days for shipment (scheduled)']


# In[48]:


df=df.drop('difference',axis=1)


# In[56]:


corr_matrix=df.corr()


# In[59]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.show()


# In[118]:


df.head()


# In[122]:





# ##### Delays are most dependant on Late risk and real days of shipping
# #### Sales have weak corr w benefit per order, mod corr w prod price, discount
# 
# 

# In[80]:


df_pred=df.select_dtypes(exclude=['object'])
df_pred.info()


# # Model Prediction

# In[61]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[81]:


X = df_pred.drop(columns=['Days for shipping (real)','Late_delivery_risk'])
y = df_pred['Delay']
print(X.shape)
print(y.shape)


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[93]:


model = RandomForestClassifier(
    n_estimators=1000
)
model.fit(X_train, y_train)


# In[97]:


y_test


# In[98]:


from sklearn import metrics
predictions = model.predict(X_test)


# In[107]:


cm = metrics.confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
# from sklearn.metrics import confusion_matrix

# # For example, considering class '0' as the positive class
# cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
# print("Confusion Matrix:\n", cm)


# In[99]:


y_test.value_counts()


# In[106]:


print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
print(f"Precision: {metrics.precision_score(y_test, predictions,average='weighted')}")
print(f"Recall: {metrics.recall_score(y_test, predictions,average='weighted')}")


# In[ ]:




