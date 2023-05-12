# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed by:NIROSHA.S
Registor No :212222230097
~~~.py
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
~~~

# OUPUT
## Dataset:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/ba7340f1-740a-4bf3-92fe-da1fc805590e)

## Tail:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/2d0e9ad0-503b-48f4-a1e2-48aec47361b1)

## Null Values:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/18a101ab-7333-4b68-8330-eeb5c4810997)

## Describe:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/f73f80d2-27b3-4404-a8d2-42b980373071)

## missing values::
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/95d54f3f-9778-4b31-bb39-97ba5f73a05f)

## Data after cleaning:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/7cd9e88a-e4e9-48dd-ae01-7aecb0c61397)

## Data on Heatmap:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/356dc323-f324-4181-aba8-589d2f3d4596)

## Report of (people survived & Died):
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/64508119-74da-4006-91ce-7c767bcf162c)

## Report of Survived People's Age:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/c0afdf2e-b2b4-486f-b6d4-a8e8c457e3b8)

## Report of pessengers:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/5a62bbd1-dddb-4995-8118-b7544ea1ea94)

## Report:
![image](https://github.com/Niroshassithanathan/Ex-07-Feature-Selection/assets/121418437/db1ca6be-c979-4e2d-a287-c33b16312446)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.












