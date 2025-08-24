import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

from config import MODEL_PATH, RANDOM_SEED
from preprocessing import SpacyPreprocessor
import pandas as pd


def load_data(path):
    """
    Expects a CSV with columns: text, label
    Uses Hugging Face datasets library
    """

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    combined_train_val_df=pd.concat((train_df,val_df),axis=0)

    return combined_train_val_df



def train(path_to_csv: str):
    dataset = load_data(path_to_csv)

    X = dataset["text"]
    y = dataset["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # return X

# Define models
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "rf": RandomForestClassifier(random_state=RANDOM_SEED),
        "xgb": XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED
        ),
    }

    pipeline = Pipeline([
        ("pre", SpacyPreprocessor()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
        ("clf", LogisticRegression())  # placeholder
    ])

    param_grid = [
        {   # Logistic Regression
            "clf": [models["logreg"]],
            "clf__C": [0.1, 1.0, 5.0],
        },
        {   # Random Forest
            "clf": [models["rf"]],
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 20],
        },
        {   # XGBoost
            "clf": [models["xgb"]],
            "clf__n_estimators": [200, 500],
            "clf__learning_rate": [0.1, 0.3],
            "clf__max_depth": [6, 10],
        },
    ]

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best model:", gs.best_estimator_)

    preds = gs.predict(X_val)
    print(classification_report(y_val, preds))
    print("Macro F1:", f1_score(y_val, preds, average="macro"))

    # Save best estimator
    joblib.dump(gs.best_estimator_, MODEL_PATH)
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    import sys
    # train(sys.argv[1])
    train("text")  # e.g. python train.py data/train.csv
      # e.g. python train.py data/train.csv
