import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score

X,y=load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)

kfold=KFold(n_splits=5, shuffle=True, random_state=42)
kfold_accuracy=cross_val_score(model,X,y,cv=kfold,scoring="accuracy")

stratified_kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_kfold_accuracy=cross_val_score(model,X,y,cv=stratified_kfold,scoring="accuracy")

loo=LeaveOneOut()
loo_accuracy=cross_val_score(model,X,y,cv=loo,scoring="accuracy")

print(f"KFold Accuracy: {kfold_accuracy}")
print(f"Stratified KFold Accuracy: {stratified_kfold_accuracy}")
print(f"Leave-One-Out Accuracy: {loo_accuracy}")

print("kfold")
print(np.mean(kfold_accuracy))
print(np.std(kfold_accuracy))