## I acknowledge the use of ChatGPT to help develop this logistic regression script
# Import libraries for data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import tools for training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Defines the features
FEATURES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Define the window size/step size for each training
EXPERIMENTS = {
    "Training_1": (50, 25),
    "Training_2": (100, 50),
    "Training_3": (200, 100)
}

# Defines the activities
ACTIVITIES = ["pushups", "situps", "upstairs", "downstairs"]

# load the dataset
df = pd.read_csv("cleaned_dataset.csv")
df = df.dropna(subset=FEATURES + ["activity", "recording"])

# Split recordings into training and testing
train_ids, test_ids = [], []

for activity, rows in df.groupby("activity"):
    recordings = rows["recording"].unique()

    if len(recordings) < 2:
        raise ValueError(f"Need at least 2 recordings for {activity}")

    train, test = train_test_split(
        recordings, test_size=0.3, random_state=42
    )

    train_ids.extend(train)
    test_ids.extend(test)

results = {}

# Run each experiment
for name, (window_size, step_size) in EXPERIMENTS.items():

    windows = []

    # Process each recording separately
    for recording, data in df.groupby("recording"):
        values = data[FEATURES].to_numpy()
        activity = data["activity"].iloc[0]

        # Divide sensor readings into overlapping windows
        for start in range(0, len(values) - window_size + 1, step_size):
            window = values[start:start + window_size]

            # Extract features from the windows
            windows.append([
                recording, activity,
                *window.mean(axis=0),
                *window.std(axis=0),
                *window.min(axis=0),
                *window.max(axis=0)
            ])

    data = pd.DataFrame(windows)

    if data.empty:
        raise ValueError(f"No complete windows for {name}")
    
    # Separate windows into training and testing datasets
    train = data[data[0].isin(train_ids)]
    test = data[data[0].isin(test_ids)]

    X_train, y_train = train.iloc[:, 2:], train[1]
    X_test, y_test = test.iloc[:, 2:], test[1]

    if train.empty or test.empty:
        raise ValueError(f"Insufficient data for {name}")

    # Creates the Logistic Regression model with feature scaling
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    )

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Displays results
    print(f"\n{name} | Window: {window_size} | Step: {step_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Calculate accuracy for each exercise
    results[name] = {}

    for activity in ACTIVITIES:
        mask = y_test == activity

        results[name][activity] = (
            accuracy_score(y_test[mask], y_pred[mask])
            if mask.any() else 0
        )

# Compares the accuracy of all three trainings
labels = ACTIVITIES
x = np.arange(len(labels))
width = 0.25

# Creates a graph
plt.figure(figsize=(10, 6))

# Creates three bars for each exercise, one for each training
for i, (name, scores) in enumerate(results.items()):
    plt.bar(
        x + (i - 1) * width,
        [scores[activity] for activity in labels],
        width,
        label=name
    )

# Adds the graph title, axis labels and exercise names
plt.title("Logistic Regression Accuracy by Exercise")
plt.xlabel("Exercise")
plt.ylabel("Accuracy")
plt.xticks(x, ["Push-ups", "Sit-ups", "Upstairs", "Downstairs"])
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()


# Saves the graph
plt.savefig("Training.png", dpi=300)

# Display the graph, if supported
if "agg" not in plt.get_backend().lower():
    plt.show()

# Closes the graph
plt.close()