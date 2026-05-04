
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sklearn
import joblib
import argparse
import os
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    print("[INFO] extracting arguments ...")

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    parser.add_argument("--train-file", type=str, default="train-V1.csv")
    parser.add_argument("--test-file", type=str, default="test-V1.csv")

    args = parser.parse_args()

    print("Sklearn version:", sklearn.__version__)
    print("Joblib version:", joblib.__version__)

    print("[INFO] reading data")

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing data")

    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    print("Column Order:", features)

    print("Training RandomForestModel")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_rep = classification_report(y_test, y_pred)

    print("\n------METRICS RESULTS FOR TESTING DATA------")
    print("Total Rows:", X_test.shape[0])
    print("[TESTING] Model Accuracy is:", test_acc)
    print("[TESTING] Classification Report is:\n", test_rep)
