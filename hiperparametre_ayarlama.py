import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


models_and_params = {
    "KNN":{
        "pipeline":Pipeline([
            ("model", KNeighborsClassifier())
        ]),
        "grid_params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__metric": ["euclidean", "manhattan"]
        },
        "random_params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__metric": ["euclidean", "manhattan"]
        }
    },
    "Decision Tree":{
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "grid_params": {
            "model__max_depth": [2,3,4,5,None],
            "model__min_samples_split": [2,4,6],
            "model__criterion": ["gini", "entropy"]
        },
        "random_params": {
            "model__max_depth": [2,3,4,5,None],
            "model__min_samples_split": [2,4,6],
            "model__criterion": ["gini", "entropy"]
        }
    },
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("model", LogisticRegression(max_iter=200, random_state=42))
        ]),
        "grid_params":{
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"]
        },
        "random_params":{
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"]
        }
    }
}

results = []


for model_name, item in models_and_params.items():
    grid_search = GridSearchCV(
        estimator=item["pipeline"],
        param_grid = item["grid_params"],
        cv = 3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_test_score = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))

    random_search = RandomizedSearchCV(
        estimator=item["pipeline"],
        param_distributions=item["random_params"],
        n_iter=4,
        cv = 3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_test_score = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))

    results.append({
        "model":model_name,
        "yontem": "grid_search",
        "cv en iyi score": round(grid_search.best_score_,2),
        "test score": round(grid_test_score, 2),
        "en iyi parametre seti": str(grid_search.best_params_) 
    })

    results.append({
        "model":model_name,
        "yontem": "random_search",
        "cv en iyi score": round(random_search.best_score_,2),
        "test score": round(random_test_score, 2),
        "en iyi parametre seti": str(random_search.best_params_) 
    })

result_df = pd.DataFrame(results)
print(result_df)