import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

x1=np.random.uniform(0,10,200)
x2=np.random.uniform(0,10,200)
x3=np.random.uniform(0,10,200)
x4=np.random.uniform(0,10,200)

X=np.column_stack((x1,x2,x3,x4))
print(X)

y=(
    4
    +2.5*x1
    +1.8*x2
    +0.15*(x1**2)
    -0.1*(x2**2)
    +0.2*(x1*x2)
    +np.random.normal(0,2,200)
)

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111, projection="3d")
ax.scatter(x1,x2,y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.title("3D Scatter Plot of Synthetic Data")
plt.show()

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

linear_model=LinearRegression()

polynomial_model=Pipeline(
    [
        ("poly",PolynomialFeatures(degree=2, include_bias=False)),
        ("linear",LinearRegression())
    ]
)

lasso_model=Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1))
    ]
)

ridge_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.1))
    ]
)

linear_model.fit(X_train, y_train)
polynomial_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

y_pred_linear=linear_model.predict(X_test)
y_pred_polynomial=polynomial_model.predict(X_test)
y_pred_lasso=lasso_model.predict(X_test)
y_pred_ridge=ridge_model.predict(X_test)


print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_linear):.4f}, R2: {r2_score(y_test, y_pred_linear):.4f}")
print(f"Polynomial Regression MSE: {mean_squared_error(y_test, y_pred_polynomial):.4f}, R2: {r2_score(y_test, y_pred_polynomial):.4f}")
print(f"Lasso Regression MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}, R2: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}, R2: {r2_score(y_test, y_pred_ridge):.4f}")

features_names=np.array(["x1","x2","x3","x4"])
lasso_coefficients=lasso_model.named_steps["lasso"].coef_

for name, coefficient in zip(features_names, lasso_coefficients):
    print(f"{name}: {coefficient:.4f}")