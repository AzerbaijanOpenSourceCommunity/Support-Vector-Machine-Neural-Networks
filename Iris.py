import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Load iris data set

iris_data = load_iris()
iris = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
iris.info()
iris.describe()

setosa_x = iris['sepal length (cm)'][:50]    #first 50 row
setosa_y = iris['sepal width (cm)'][:50]
versicolor_x = iris['sepal length (cm)'][50:100]
versicolor_y = iris['sepal width (cm)'][50:100]

# Visaulizing our data setosa and versicolor


plt.title("Iris data set visualization")
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.scatter(setosa_x, setosa_y, marker='*', color='green')
plt.scatter(versicolor_x, versicolor_y, marker='+', color='red')
#

#
from sklearn.model_selection import train_test_split

X = iris
Y = iris_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=150)
from sklearn.svm import SVC

model = SVC(gamma=0.01, kernel='linear', C=100)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score


## Calculating  precision

print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions,target_names=iris_data.target_names))


plt.show()
