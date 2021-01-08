import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_data=pd.read_csv('student/student-mat.csv')

data=_data.drop('school', axis=1)


from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
train_labels=data['G3'][:316]

test_labels=test_set['G3']

train_set=train_set.drop('G3', axis=1)
test_set=test_set.drop('G3',axis=1)


#Classifier

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(train_set, train_labels)
pred=clf.predict(test_set)
pred


# ## Predict to myself

labels=[1,16,0,1,0,4,4,0,0,1,1,4,2,0,1,0,0,0,0,1,1,4,2,1,0,1,1,4,0,15,15]
labels=np.resize(labels, (1,31))



clf.predict(labels)


# ## Error Measurement

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(test_labels,pred )
rmse=np.sqrt(mse)


# ## Exporting Model

from joblib import dump, load
dump(clf, 'StudentsProgressModel.joblib')


