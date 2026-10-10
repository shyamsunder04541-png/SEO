import pandas as pd
import pickle
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. Find all YouTube CSV files
# ==========================================

files = glob.glob("youtube-spam-collection-v1/*.csv")

print("Files found:")
for file in files:
    print(file)


# ==========================================
# 2. Read all CSV files
# ==========================================

dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all 5 datasets
df = pd.concat(dataframes, ignore_index=True)


print("\nTotal comments:", len(df))
print("Columns:", df.columns.tolist())


# ==========================================
# 3. Select required columns
# ==========================================

df = df[["CONTENT", "CLASS"]]

# Remove empty comments
df = df.dropna()


# ==========================================
# 4. Convert labels
# ==========================================

# 1 = Spam
# 0 = Not Spam

df["CLASS"] = df["CLASS"].map({
    1: "spam",
    0: "not_spam"
})


# ==========================================
# 5. Input and output
# ==========================================

X = df["CONTENT"]
y = df["CLASS"]


# ==========================================
# 6. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining comments:", len(X_train))
print("Testing comments:", len(X_test))


# ==========================================
# 7. Create Machine Learning Pipeline
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),

    (
        "classifier",
        MultinomialNB()
    )
])


# ==========================================
# 8. Train model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)


# ==========================================
# 9. Test model
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 10. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print(accuracy)


# ==========================================
# 11. Classification Report
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# 12. Confusion Matrix
# ==========================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# 13. Save trained model
# ==========================================

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print("spam_model.pkl")