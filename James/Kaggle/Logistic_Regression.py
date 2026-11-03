## I acknowledge the use of ChatGPT to help develop this logistic regression script
# Import libraries for data processing
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import tools for training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Define the window size/step size for each training
EXPERIMENTS = {
    "Training_1": (50, 25),
    "Training_2": (100, 50),
    "Training_3": (200, 100)
}

# Dataset folder names
ACTIVITIES = {
    "9.Push-up": "pushups",
    "10.Sit-up": "situps",
    "15.Stair-up": "upstairs",
    "16.Stair-down": "downstairs"
}

# Find recordings in dataset
recordings = []

for path in Path("2.Trimmed_interpolated_data").rglob("*.csv"):
    for folder, activity in ACTIVITIES.items():
        if folder in path.parts:
            recordings.append((path, activity))
            break

if not recordings:
    raise ValueError("No KU-HAR recordings found.")

# Split recordings into training and testing
train_files = set()

for activity in ACTIVITIES.values():
    files = [path for path, label in recordings if label == activity]

    if len(files) < 2:
        raise ValueError(f"Not enough recordings for {activity}")

    train, test = train_test_split(
        files, test_size=0.3, random_state=42
    )

    train_files.update(train)

results = {}

# Run all three trainings
for name, (window_size, step_size) in EXPERIMENTS.items():

    # Creates empty lists for training and testing data
    X_train, X_test, y_train, y_test = [], [], [], []

    for path, activity in recordings:

        # Reads the accelerometer and gyroscope data
        values = pd.read_csv(
            path, header=None, usecols=[1, 2, 3, 5, 6, 7]
        ).apply(pd.to_numeric, errors="coerce").dropna().to_numpy()

        # Assign accelerometer and gyroscope data to training or testing
        X, y = (X_train, y_train) if path in train_files else (X_test, y_test)

        # Divide readings into overlapping windows
        for start in range(0, len(values) - window_size + 1, step_size):
            window = values[start:start + window_size]

            # Extract features
            X.append([
                *window.mean(axis=0),
                *window.std(axis=0),
                *window.min(axis=0),
                *window.max(axis=0)
            ])

            y.append(activity)

    if not X_train or not X_test:
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
    predictions = model.predict(X_test)

    # Displays results
    print(f"\n{name} | Window: {window_size} | Step: {step_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    # Calculate accuracy for each exercise
    results[name] = {}

    y_test = np.array(y_test)

    for activity in ACTIVITIES.values():
        mask = y_test == activity

        results[name][activity] = (
            accuracy_score(y_test[mask], predictions[mask])
            if mask.any() else 0
        )

# Compares the accuracy of all three trainings
labels = list(ACTIVITIES.values())
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


# Save the graph
plt.savefig("Training.png", dpi=300)


# Display the graph if Matplotlib supports it
if "agg" not in plt.get_backend().lower():
    plt.show()

# Closes the Graph
plt.close()
