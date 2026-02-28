import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

DATA_PATH = "data/data.xlsx"
SHEET_NAME = "Vehicle_Maintenance_Data"
MODEL_PATH = "model/model_bundle.joblib"
META_PATH = "model/metadata.json"

TARGET_CLASS = "Component_Category"
TARGET_COST = "Total_Cost"

DROP_COLUMNS = [
    "Vehicle_ID",
    "Specific_Component_Repaired"
]

def main():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.drop(columns=DROP_COLUMNS)

    X = df.drop(columns=[TARGET_CLASS, TARGET_COST])
    y_class = df[TARGET_CLASS]
    y_cost = df[TARGET_COST]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols
            ),
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median"))
                ]),
                numeric_cols
            )
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"
    )

    reg = RandomForestRegressor(
        n_estimators=400,
        random_state=42
    )

    class_pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    cost_pipe = Pipeline([("prep", preprocessor), ("model", reg)])

    X_train, X_test, yc_train, yc_test, ycost_train, ycost_test = train_test_split(
        X, y_class, y_cost,
        test_size=0.25,
        random_state=42,
        stratify=y_class
    )

    class_pipe.fit(X_train, yc_train)
    cost_pipe.fit(X_train, ycost_train)

    class_preds = class_pipe.predict(X_test)
    cost_preds = cost_pipe.predict(X_test)

    acc = accuracy_score(yc_test, class_preds)
    mae = mean_absolute_error(ycost_test, cost_preds)
    r2 = r2_score(ycost_test, cost_preds)

    os.makedirs("model", exist_ok=True)

    joblib.dump(
        {"class_pipe": class_pipe, "cost_pipe": cost_pipe},
        MODEL_PATH
    )

    metadata = {
        "features": list(X.columns),
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "metrics": {
            "classification_accuracy": float(acc),
            "cost_mae": float(mae),
            "cost_r2": float(r2)
        }
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model trained successfully")

if __name__ == "__main__":
    main()
