import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dat = pd.read_csv("HRDataset_v9.csv")
print(dat.shape)
dat.head(3)

numcols = ['Age', 'Pay Rate', 'Zip', 'Days Employed']
catcols = ['Sex']
data = dat[numcols+catcols]

data.describe()

Y = data['Days Employed']>=1238
X = data.copy()
del X['Days Employed']
len(X.columns)


from sklearn.model_selection import train_test_split

X = data.iloc[:,1:3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 10, random_state = seed)
kfold

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

cv_logistic_reg = cross_val_score(LogisticRegression(),
                                 X_train, Y_train, cv=kfold, scoring = 'accuracy')
cv_logistic_reg



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = []
models.append(( 'LR' , LogisticRegression()))
models.append(( 'LDA' , LinearDiscriminantAnalysis()))
models.append(( 'KNN' , KNeighborsClassifier()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=10, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring= 'accuracy' )
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

predictions


hasil = X_validation
hasil["label_predict"] = predictions
hasil["actual_label"] = Y_validation
hasil[hasil["label_predict"] != hasil["actual_label"]]

hasil
