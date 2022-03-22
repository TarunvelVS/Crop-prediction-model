#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact


# In[2]:


A_data = pd.read_csv('Agri_Data.csv')


# In[3]:


A_data.shape


# In[4]:


A_data.head()


# In[70]:


A_data.info()


# In[6]:


A_data.isnull().sum()


# In[8]:


A_data['label'].value_counts()


# In[10]:


A_data['N'].mean()


# In[11]:


A_data['P'].mean()


# In[12]:


A_data['K'].mean()


# In[13]:


A_data['temperature'].mean()


# In[14]:


A_data['humidity'].mean()


# In[15]:


A_data['ph'].mean()


# In[16]:


A_data['rainfall'].mean()


# In[19]:


@interact
def summary(crops=list(A_data['label'].value_counts().index)):
    x=A_data[A_data['label']==crops]
    print("Stats for Nitogen")
    print("Min Req :",x['N'].min())
    print("Average Req :",x['N'].mean())
    print("Max Req :",x['N'].max())
    print("/n")

    print("Stats for Phosphorous")
    print("Min Req :",x['P'].min())
    print("Average Req :",x['P'].mean())
    print("Max Req :",x['P'].max())
    print("/n")
    
    print("Stats for Potassium")
    print("Min Req :",x['K'].min())
    print("Average Req :",x['K'].mean())
    print("Max Req :",x['K'].max())
    print("/n")
    
    print("Stats for Temperature")
    print("Min Req :",x['temperature'].min())
    print("Average Req :",x['temperature'].mean())
    print("Max Req :",x['temperature'].max())
    print("/n")
    
    print("Stats for Humidity")
    print("Min Req :",x['humidity'].min())
    print("Average Req :",x['humidity'].mean())
    print("Max Req :",x['humidity'].max())
    print("/n")
    
    print("Stats for ph")
    print("Min Req :",x['ph'].min())
    print("Average Req :",x['ph'].mean())
    print("Max Req :",x['ph'].max())
    print("/n")
    
    print("Stats for rainfall")
    print("Min Req :",x['rainfall'].min())
    print("Average Req :",x['rainfall'].mean())
    print("Max Req :",x['rainfall'].max())
    print("/n")


# ### Distribution

# In[69]:


import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20,9)

plt.subplot(2,4,1)
sns.distplot(A_data['N'])
plt.xlabel('Ratio of Nitrogen')
plt.grid()

plt.subplot(2,4,2)
sns.distplot(A_data['P'])
plt.xlabel('Ratio of Phosphorous')
plt.grid()

plt.subplot(2,4,3)
sns.distplot(A_data['K'])
plt.xlabel('Ratio of Potassium')
plt.grid()

plt.subplot(2,4,4)
sns.distplot(A_data['temperature'])
plt.xlabel('Ratio of Temperature')
plt.grid()

plt.subplot(2,4,5)
sns.distplot(A_data['humidity'])
plt.xlabel('Ratio of Humidity')
plt.grid()

plt.subplot(2,4,6)
sns.distplot(A_data['ph'])
plt.xlabel('Ratio of ph')
plt.grid()

plt.subplot(2,4,7)
sns.distplot(A_data['rainfall'])
plt.xlabel('Ratio of rainfall')
plt.grid()


# ### Analysis of Patterns in the data set

# In[25]:


# Crops requiring higher nitrogen contents

A_data[A_data['N']>100]['label'].unique()


# In[26]:


# crops requiring lesser nitrogen contents

A_data[A_data['N']<25]['label'].unique()


# In[27]:


# crops requiring higher phosphorus
A_data[A_data['P']>100]['label'].unique()


# In[28]:


# crops requiring higher Pottasium content
A_data[A_data['K']>100]['label'].unique()


# In[30]:


# High rainfall requiring crops
A_data[A_data['rainfall']>200]['label'].unique()


# In[33]:


# High temperature requiring crops

A_data[A_data['temperature']>38]['label'].unique()


# In[34]:


# low humidity 

A_data[A_data['humidity']<40]['label'].unique()


# In[35]:


# low pH

A_data[A_data['ph']<3.5]['label'].unique()


# In[36]:


# low pH

A_data[A_data['ph']<4]['label'].unique()


# In[40]:


# Summer Crops

A_data[(A_data['humidity']>50) & (A_data['temperature']>35)]['label'].unique()


# In[41]:


# Winter Crops

A_data[(A_data['humidity']>30) & (A_data['temperature']<20)]['label'].unique()


# In[42]:


# Rainy Crops

A_data[(A_data['humidity']>30) & (A_data['rainfall']>200)]['label'].unique()


# ## Model Creation

# In[44]:


from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

x=A_data.loc[:,['N','P','K','temperature','ph','humidity','rainfall']].values

x.shape


# In[46]:


x_data = pd.DataFrame(x)
x_data.head()


# In[47]:


plt.rcParams['figure.figsize'] = (10,4)

list_1 = []

for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    list_1.append(km.inertia_)
    
plt.plot(range(1,11),list_1)
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('list_1')
plt.show()


# In[49]:


km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a=A_data['label']
y_means=pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns = {0:'cluster'})

z[z['cluster']==0]['label'].unique()


# In[50]:


z[z['cluster']==1]['label'].unique()


# In[51]:


z[z['cluster']==2]['label'].unique()


# In[52]:


z[z['cluster']==3]['label'].unique()


# In[53]:


z[z['cluster']==4]['label'].unique()


# In[54]:


y=A_data['label']
x=A_data.drop(['label'],axis=1)

x.shape


# In[55]:


y.shape


# In[56]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape


# In[57]:


y_train.shape


# In[58]:


x_test.shape


# In[59]:


y_test.shape


# In[60]:


# Predictive Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[62]:


from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)
print(cr)


# ## Results

# In[64]:


prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))
print("Suggested Crop for above mentioned condition:", prediction)


# In[66]:


A_data.head()


# In[67]:


A_data[A_data['label']=='coconut'].head()


# In[68]:


prediction = model.predict((np.array([[20,30,29,25,90,6,180]])))
print("Suggested Crop for above mentioned condition:", prediction)


# In[ ]:




