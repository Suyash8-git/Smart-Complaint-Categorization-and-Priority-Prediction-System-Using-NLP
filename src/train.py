"""Train + evaluate. Run from project root:  python -m src.train"""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocessing import tokenize


def make_pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", clf),
    ])


def evaluate(name, pipe, X_te, y_te, out):
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    out.append(f"\n===== {name} =====\nAccuracy: {acc:.4f}\n{classification_report(y_te, pred, zero_division=0)}")
    return acc, pred


def save_cm(y_te, pred, labels, title, path):
    cm = confusion_matrix(y_te, pred, labels=labels)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.xticks(rotation=40, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    df = pd.read_csv("data/complaints.csv")
    X = df["complaint"]
    report = [f"Dataset: {len(df)} complaints"]

    # ---- Category: compare Naive Bayes vs Logistic Regression
    Xtr, Xte, ytr, yte = train_test_split(X, df["category"], test_size=0.2, stratify=df["category"], random_state=42)
    candidates = {"Logistic Regression": make_pipe(LogisticRegression(C=10, max_iter=2000)), "Naive Bayes": make_pipe(MultinomialNB())}  # LR first: wins ties
    scores = {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        scores[name], _ = evaluate(f"CATEGORY - {name}", pipe, Xte, yte, report)
    best = max(scores, key=scores.get)
    report.append(f"\nModel comparison (category): {scores}\nSelected: {best}")
    best_pipe = candidates[best]
    save_cm(yte, best_pipe.predict(Xte), sorted(df.category.unique()), f"Category - {best}", "reports/confusion_matrix_category.png")
    joblib.dump(best_pipe, "models/category_model.pkl")

    # ---- Priority
    order = ["Low", "Medium", "High", "Critical"]
    Xtr, Xte, ytr, yte = train_test_split(X, df["priority"], test_size=0.2, stratify=df["priority"], random_state=42)
    pri = make_pipe(LogisticRegression(C=20, max_iter=3000, class_weight="balanced"))
    pri.fit(Xtr, ytr)
    evaluate("PRIORITY - Logistic Regression", pri, Xte, yte, report)
    save_cm(yte, pri.predict(Xte), order, "Priority - Logistic Regression", "reports/confusion_matrix_priority.png")
    joblib.dump(pri, "models/priority_model.pkl")

    open("reports/metrics.txt", "w").write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
