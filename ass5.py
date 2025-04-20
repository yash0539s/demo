import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from ydata_profiling import profile_report

data=fetch_openml('titanic',version=1)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

df.head()
df.info()
df.describe()
df.isnull().sum()
df['target'].value_counts()
print(df['target'].value_counts(normalize=True))
sns.countplot(x='target',data=df)
plt.title('Count of Survived vs Not Survived')
plt.show()
sns.histplot(df['age'],bins=30,kde=True)
plt.title('Age Distribution')
plt.show()
sns.boxplot(x='target',y='age',data=df)
plt.title('Boxplot of Age by Survival')
plt.show()

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')    
plt.show()

profile = profile_report(df, title="Titanic Data EDA Report")
profile.to_file("titanic_eda_report.html")