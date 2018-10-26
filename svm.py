import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from scipy import stats

cancer = load_breast_cancer()
print(cancer.keys())

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names']) # our dataframe
from sklearn.model_selection import train_test_split

a = df
b = cancer['target']    # target
a_train, a_test, b_train, b_test = train_test_split( a, b, test_size=0.33, random_state=40)
from sklearn.svm import SVC
model = SVC(gamma=0.001,kernel='rbf',C=1000)
model.fit(a_train,b_train)
predictions = model.predict(a_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(b_test,predictions))
print(classification_report(b_test,predictions))

