from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

heart_disease = fetch_ucirepo(id=45)

df=pd.DataFrame(data=heart_disease.data.features)
df["target"]=heart_disease.data.targets
df["target"]=df["target"].apply(lambda x: 0 if x==0 else 1)

print(df.head())

if df.isna().any().any():
    df.dropna(inplace=True)
    print("Dropped rows with missing values.")
else:
    print("No missing values found.")

X=df.drop(["target"],axis=1).values
y=df.target.values

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,random_state=42)


log_reg=LogisticRegression(penalty="l2",C=1.0,max_iter=100)
log_reg.fit(X_train,y_train)

acc=log_reg.score(X_test,y_test)
print(f"Accuracy of Logistic Regression model: {acc:.4f}")