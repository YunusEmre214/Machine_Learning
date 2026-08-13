from sklearn.datasets import load_breast_cancer # dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

import pandas as pd
import matplotlib.pyplot as plt

cancer=load_breast_cancer()
df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target
print(df.head())

X=cancer.data
y=cancer.target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled,y_train)

y_pred=knn.predict(X_test_scaled)

acc=accuracy_score(y_test,y_pred)
print(f"Accuracy of KNN model: {acc:.4f}")

conf_matrix=confusion_matrix(y_test,y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

k_accuracy = []
k_values = []
for k in range(3, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    k_accuracy.append(accuracy_score(y_pred, y_test))
    k_values.append(k)

plt.plot(k_values, k_accuracy)
plt.show()