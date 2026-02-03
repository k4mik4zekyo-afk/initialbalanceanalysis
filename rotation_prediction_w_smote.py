'''
Work log Jan 30th
-Probably will want to investigate the 10 or so dropped data points
-Full model has an 84% accuracy for rotation
-Cross validation is somewhat poor 0.42
-Feature engineer time in which the rotation occurs:
    #If using bar #
    df['rotation_time_normalized'] = rotation_bar / df['bar_number'].max()
-Could there be more opportunities with the confluence columns?


'''

import pandas as pd
import sys
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle

def classify_rotation(row):
    if row["rotation"] and row["breakside"] == "low":
        return "rotates_up"
    elif row["rotation"] and row["breakside"] == "high":
        return "rotates_down"
    else:
        return "non_rotation"

# Load your data
df = pd.read_csv("phase2_previous_day_levels.csv")  # Update with your file path
# Desktop/LearningPython/InitialBalance_Analytics/phase2_previous_day_levels.csv

#Rotation class
df["rotation_direction"] = df.apply(classify_rotation, axis=1)

# Normalize nearest_prior_level_to_open_distance by previous day's range (PDH - PDL)
df["prev_range"] = df["prev_pdh"] - df["prev_pdl"]

# Handle divide-by-zero safely
df["prev_range"] = df["prev_range"].replace(0, np.nan)  # avoid div-by-zero
df["normalized_distance"] = df["nearest_prior_level_to_open_distance"] / df["prev_range"]

# Fill any NaNs from division (e.g., flat previous range) with 0 or an appropriate fallback
df["normalized_distance"] = df["normalized_distance"].fillna(0)

label_encoder = LabelEncoder()
#df["rotation_direction_encoded"] = label_encoder.fit_transform(df["rotation_direction"])

# Convert True/False to string labels
df["rotation_str"] = df["rotation"].map({True: "Rotation", False: "Non-Rotation"})

# Then encode as numbers
label_encoder = LabelEncoder()
df["rotation_encoded"] = label_encoder.fit_transform(df["rotation_str"])

# %% Define features and target
#When IB is established
X = df[["relative_ib_volume", "normalized_distance", "opening_bar_open_close",
        "opening_bar_volume", "prev_session_volume"]]
#At opening
#X = df[["prev_range", "prev_session_volume", "opening_bar_open_close", "opening_bar_volume"]]

#y = df["rotation_direction_encoded"]
y = df["rotation"]

# Drop rows with NaNs in features
X = X.dropna()
y = y.loc[X.index]

# %% Applying SMOTE to a single decision tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Split the original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 2: Apply SMOTE only to training data
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train,y_train)

# Step 3: Create model and train on SMOTE data
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Step 4a: Evaluate on original test data
score = model.score(X_test, y_test)
print(f"Test Accuracy: {score}")

# Step 4b: Evaluate on original train data
score_b = model.score(X_train_smote, y_train_smote)
print(f"Train Accuracy: {score_b}")

# Step 4c: Export the model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Step 5: Visualize the decision tree
plt.figure(figsize=(16, 10))
plot_tree(model,
          feature_names=X.columns,
          class_names=label_encoder.classes_,
          filled=True,
          proportion=True,
          rounded=True)
plt.title("Decision Tree")
plt.savefig('Decision Tree proportions.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: Make a prediction using new data | TESTING

#[["relative_ib_volume", "normalized_distance", "opening_bar_open_close",
#        "opening_bar_volume", "prev_session_volume"]]
var1=0.301 #relative_ib_volume
var2=10 #normalized_distance
var3=-10 #opening_bar_open_close
var4=9000 #opening_bar_volume
var5=999 #prev_session_volume
#should result in rotation

new_data = [[var1, var2, var3, var4, var5]]
new_data = [[0.3118, 0.9284, -21.5, 14873.0, 1787488.0]]

prediction = model.predict(new_data)
print(prediction)

# %% What if - non-rotations


# %% Cross validation - sandbox
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE

# # Define your pipeline
# pipeline = Pipeline(steps=[
#     ("smote", SMOTE(random_state=42)),
#     ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
# ])

# # Define fold strategies to test
# fold_configs = {
#     "5-Fold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
#     "10-Fold": StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# }

# # Store results
# results = {}

# # Run each CV strategy and collect scores
# for label, cv in fold_configs.items():
#     scores = cross_val_score(pipeline, X, y, scoring="f1", cv=cv)
#     results[label] = {
#         "Scores": scores,
#         "Average F1": scores.mean(),
#         "Std Dev": scores.std()
#     }

# # Display results
# print("Cross-Validation Comparison (F1 Score):\n")
# for label, data in results.items():
#     print(f"{label}:")
#     print(f"  Scores: {data['Scores']}")
#     print(f"  Avg F1: {data['Average F1']:.3f}")
#     print(f"  Std Dev: {data['Std Dev']:.3f}")
#     print("-" * 40)

# sys.exit()

# %% Archive the old version model results are balanced 80% accuracy
# Use the resampled data instead of original X, y
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
)

#clf = RandomForestClassifier(n_estimators=200, random_state=42)

#Use this if you want to balance the 'uneven' rotation class
clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42
    )

clf.fit(X_train, y_train)

# Evaluate a multiclass model
y_pred = clf.predict(X_test)

#print(classification_report(y_test, y_pred, target_names=label_encoder.classes_)) #Use for multiclass
print(classification_report(y_test, y_pred, target_names=["Non-Rotation", "Rotation"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

sys.exit()

# %% Visualize the errors - detailed
# Predict on the test set
y_pred = clf.predict(X_test)

# Create a DataFrame to compare predictions with actual labels
results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred

# Filter for misclassified points
misclassified = results[results["actual"] != results["predicted"]]

# Show or save the misclassified results
print(misclassified)
#['non-rotation', 'rotation_down', 'rotation_up']

sys.exit()

# %% Feature Importances

import pandas as pd

feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))

# %% Visualize one tree

from sklearn.tree import plot_tree

# Pick the first tree
plt.figure(figsize=(16, 10))
plot_tree(clf.estimators_[0], 
          feature_names=X.columns, 
          class_names=label_encoder.classes_, 
          filled=True, 
          rounded=True)
plt.title("One Tree from Random Forest")
plt.show()
sys.exit()

# %%% Plot the decision tree with percentages instead of counts, because SMOTE
# inflates the count

from sklearn.tree import plot_tree
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree.fit(X_train, y_train)

# plt.figure(figsize=(16, 10))
# plot_tree(clf_tree, 
#           feature_names=X.columns,
#           class_names=label_encoder.classes_,
#           filled=True,
#           rounded=True)
# plt.title("Decision Tree")
# plt.show()

plt.figure(figsize=(16, 10))
plot_tree(clf_tree,
          feature_names=X.columns,
          class_names=label_encoder.classes_,
          filled=True,
          proportion=True,
          rounded=True)
plt.title("Decision Tree")
#plt.show()
plt.savefig('Decision Tree proportions.png')


# %% Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming y_test (true labels) and y_pred (model predictions)
cm = confusion_matrix(y_test, y_pred)

# Display it visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# %% Quick data mining

#Count unique occurrences of rotations in df data set (True/False)
print(df.rotation.value_counts())