import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("core_dataset.csv")
print(df.shape)
df.head(2)


#to analyse the Employee Source
sns.set(rc={'figure.figsize':(7,7)})
sns.countplot(y="Employee Source", data=df)

#to analyse the Employee 
print("Number of employes: "+str(len(df.index)))
sns.countplot(x="MaritalDesc", hue="Sex", data=df)

#to see the age distribution
df["Age"].plot.hist()
