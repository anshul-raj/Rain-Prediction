import pandas as pd
import numpy as np
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC



DF = pd.read_csv("/Users/ccs/Anshul-Raj/ML-project/Dataset/weatherAUS.csv")
DF = DF.drop(columns = ['RISK_MM'])
columns = list(DF.columns)

for c in columns:
  print('null count in',c,'=',DF[c].isnull().sum())

DF = DF.dropna(subset=['RainToday'])
c = 'RainToday'

print('null count in',c,'=',DF[c].isnull().sum())

DF = DF.replace(to_replace={'Yes':1,'No':0})
# -------------------

rain_true = DF[DF['RainTomorrow']==1]
rain_false = DF[DF['RainTomorrow']==0]
rain_false = rain_false.sample(n=len(rain_true),random_state = 0)

DF = pd.concat([rain_true,rain_false],axis=0,ignore_index=True).reset_index(drop=True)
DF.columns = columns
for i in [1,0]:
    print("count of",i,"=",len(DF[DF.RainTomorrow==i]))
print(DF)

DF.to_csv(r"./balanced_datased.csv",sep=",",index=False)

# -------------------
y = DF[['RainTomorrow']]
DF=DF.drop(columns = ['RainTomorrow'])

numerical_col = [i for i in DF.columns if (DF[i].dtype=='float64' or DF[i].dtype=='int64')]
objects_col = [i for i in DF.columns if DF[i].dtype=='object']

print(numerical_col)
print(objects_col)

X_train, X_test, y_train, y_test = train_test_split(DF,y, test_size=0.2, stratify=y, random_state=0)
t = X_train.median()
X_train = X_train.fillna(t)
X_test = X_test.fillna(t)

for t_df in [X_train, X_test]:
    t_df['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    t_df['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    t_df['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    t_df['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

X_train = pd.concat([X_train[numerical_col], pd.get_dummies(X_train.Location,prefix='Location'),
                    pd.get_dummies(X_train.WindGustDir,prefix='WindGustDir'),
                    pd.get_dummies(X_train.WindDir9am,prefix='WindDir9am'),
                    pd.get_dummies(X_train.WindDir3pm,prefix='WindDir3pm'),
                    pd.get_dummies(X_train.RainToday,prefix='RainToday')], axis=1)

X_test = pd.concat([X_test[numerical_col], pd.get_dummies(X_test.Location,prefix='Location'),
                    pd.get_dummies(X_test.WindGustDir,prefix='WindGustDir'),
                    pd.get_dummies(X_test.WindDir9am,prefix='WindDir9am'),
                    pd.get_dummies(X_test.WindDir3pm,prefix='WindDir3pm'),
                    pd.get_dummies(X_test.RainToday,prefix='RainToday')], axis=1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


for i in [1,0]:
    print("count of",i,"=",len(y_test[y_train==i]))

