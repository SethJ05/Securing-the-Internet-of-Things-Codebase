ty"].shift()).cumsum()

means = df.groupby(block)[sensors].transform(
    lambda x: x.rolling(50, min_periods=2).mean()
).fillna(0)
means.columns = [col + "_mean" for col in sensors]

stds = df.groupby(block)[sensors].transform(
    lambda x: x.rolling(50, min_periods=2).std()
).fillna(0)
stds.columns = [col + "_std" for col in sensors]

df = pd.concat([df, means, stds], axis=1)
features = sensors + means.columns.tolist() + stds.columns.tolist()

# Test/Train split.
splitter = GroupShuffleSplit(test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["activity"]))
train_data = df.iloc[train_idx]
test_data = df.iloc[test_idx]

# Use an equal number of readings from each exercise.
train_data = train_data.groupby(target, group_keys=False).sample(
    n=7500, random_state=42
)
test_data = test_data.groupby(target, group_keys=False).sample(
    n=1000, random_state=42
)

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train model, setting n_neighbours as 5 yielded the highest results
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print model performance.
print("Training readings:", len(X_train))
print("Testing readings:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=3))

# Print confusion matrix.
classes = sorted(df[target].unique())
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=classes,
    display_labels=classes, cmap="Blues", values_format="d"
)
plt.title("KNN exercise classification: confusion matrix")
plt.tight_layout()
plt.savefig(folder / "knn_confusion_matrix.png", dpi=160)
plt.show()

# Graph precision, recall and F1-score for each exercise.
report = classification_report(y_test, y_pred, output_dict=True)
by_exercise = pd.DataFrame(report).T.loc[
    classes, ["precision", "recall", "f1-score"]
]
ax = by_exercise.plot(kind="bar", figsize=(9, 5), rot=0)
ax.set_title("KNN scores by exercise")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(folder / "knn_metrics_graph.png", dpi=160)
plt.show()

print("\nSaved two charts in the project folder.")

