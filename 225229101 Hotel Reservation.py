#!/usr/bin/env python
# coding: utf-8

# ### Import Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Hotel Reservations.csv")
df.head()


# **Cleaning the data make it numberic value**

# In[3]:


data = df.replace({"type_of_meal_plan":{"Not Selected":0,"Meal Plan 1":1,"Meal Plan 2":2, "Meal Plan 3":3},
                        "room_type_reserved":{"Room_Type 1":1,"Room_Type 2":2,"Room_Type 3":3,"Room_Type 4":4,"Room_Type 5":5,"Room_Type 6":6,"Room_Type 7":7},
                        "market_segment_type":{"Offline":0,"Online":1, "Corporate":3,"Aviation": 4,"Complementary":5},
                        "booking_status":{"Canceled":0,"Not_Canceled":1}})


# In[4]:


data.info()


# In[5]:


data.describe()


# In[7]:


data.drop(["booking_status"],axis=1).corrwith(data["booking_status"])


# **Build the data training and Test Set**

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = data.drop(["Booking_ID","arrival_month","arrival_date", "booking_status"],axis=1)
y = data["booking_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# **Make model pipeline**

# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

model_pipeline = []
model_pipeline.append(LogisticRegression(solver='liblinear'))
model_pipeline.append(SVC())
model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(DecisionTreeClassifier())
model_pipeline.append(RandomForestClassifier())
model_pipeline.append(GaussianNB())


# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model_list = ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest", "Naive Bayes"]
acc_list =[]
auc_list = []
cm_list = []

for model in model_pipeline:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_list.append(metrics.accuracy_score(y_test,y_pred))
    fpr, tpr ,_tresholds = metrics.roc_curve(y_test,y_pred)
    auc_list.append(round(metrics.auc(fpr,tpr),2))
    cm_list.append(confusion_matrix(y_test,y_pred))


# **Make heatmap result from test data set**

# In[ ]:


fig = plt.figure(figsize=(18,10))
for i in range(len(cm_list)):
    cm = cm_list[i]
    model= model_list[i]
    sub= fig.add_subplot(2,3, i+1).set_title(model)
    cm_plot = sns.heatmap(cm, annot=True, cmap="Blues_r")
    cm_plot.set_xlabel("Predicted Values")
    cm_plot.set_ylabel("Actual Values")


# **Finding the Best to find the best model**

# In[ ]:


result_df = pd.DataFrame({"Model":model_list,"Accuracy" :acc_list, "AUC":auc_list })
result_df


# the result say that Random Forest are the best model for the data set to make predicion

# Make a prediction

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[ ]:


#checking the accuracy pf the model
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# **Preparing the data, here I use data from the updateed version of the original data**

# In[ ]:


guest = 36273 # can change by the guest Booking ID
data1=data.iloc[[guest]].drop(["Booking_ID","arrival_month","arrival_date", "booking_status"],axis=1)
#limiting the input data cause random forest just eccept 15 feature. so I eliminate by the less correlated to booking_status
data1.values.tolist()


# In[ ]:


code = clf.predict(data)
if code == 0:
    print("Not Cancel")
else:
    print("cancel")


# In[ ]:




