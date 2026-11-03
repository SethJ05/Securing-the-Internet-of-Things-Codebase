from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# Loading dataset

folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "cleaned_dataset(2).csv")

sensors = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

target = "recording"


# After researching improvement methods I came across 'Rolling' this has been implemented below.

# Keep recording sequences separate
block = df["activity"].ne(df["activity"].shift()).cumsum()

# Calculate rolling averages of the previous 50 readings
means = df.groupby(block)[sensors].transform(
    lambda x: x.rolling(50, min_periods=2).mean()
).fillna(0)

means.columns = [col + "_mean" for col in sensors]

# Calculate rolling standard deviations
stds = df.groupby(block)[sensors].transform(
    lambda x: x.rolling(50, min_periods=2).std()
).fillna(0)

stds.columns = [col + "_std" for col in sensors]

# Add the new features to the dataset
df = pd.concat([df, means, stds], axis=1)

features = (
    sensors
    + means.columns.tolist()
    + stds.columns.tolist()
)


# Train/Test Split

splitter = GroupShuffleSplit(
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["activity"])
)

train_data = df.iloc[train_idx]
test_data = df.iloc[test_idx]

# Balance the number of readings per exercise
train_data = train_data.groupby(
    target, group_keys=False
).sample(n=7500, random_state=42)

test_data = test_data.groupby(
    target, group_keys=False
).sample(n=1000, random_state=42)


# Separate features and target labels
X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


#  Train KNN model

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )
)

model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)


# Display performance metrics

print("Training readings:", len(X_train))
print("Testing readings:", len(X_test))

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))


# Confusion matrix

classes = sorted(df[target].unique())

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=classes,
    display_labels=classes,
    cmap="Blues",
    values_format="d"
)

plt.title("KNN Exercise Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png", dpi=160)
plt.show()


# Precision, Recall & F1 score

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics = pd.DataFrame(report).T.loc[
    classes,
    ["precision", "recall", "f1-score"]
]

metrics.plot(
    kind="bar",
    figsize=(9, 5),
    rot=0
)

plt.title("KNN Performance by Exercise")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("knn_performance_metrics.png", dpi=160)
plt.show()
