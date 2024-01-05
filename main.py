import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("./archive.zip")
sns.countplot(x=df['Sex'], hue=df['Survived'])
df.groupby('Sex')[['Survived']].mean()
df['Sex'].unique()
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])
X= df[['Pclass', 'Sex']]
Y=df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)

pred = print(log.predict(X_test))

import warnings
warnings.filterwarnings("ignore")

res= log.predict([[2,1]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
