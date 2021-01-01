import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)

# Import the dataset
df = pd.read_csv("I:/archive/understat_per_game.csv")
df.head()
# Create subsets by filtering the original dataframe
bundesliga = df[df["league"] == "Bundesliga"]
leipzig = bundesliga[bundesliga["team"] == "RasenBallsport Leipzig"]
# Convert the home/away values into numerical equivalents so that mathematical concepts could be applied
bundesliga["h_a_n"] = bundesliga.h_a.replace({"h": 1, "a": 0})
leipzig["h_a_n"] = leipzig.h_a.replace({"h": 1, "a": 0})
# Select features for the model
features = bundesliga[["xG", "h_a_n", "deep", "ppda_coef", "oppda_coef"]]
features_leipzig = leipzig[["xG", "h_a_n", "deep", "ppda_coef", "oppda_coef"]]
# Select the response we want predicted
target = bundesliga["result"]
target_leipzig = leipzig["result"]
target = target.sort_values()
# See if there are any issues
leipzig.head()

# Create an empty list to store the different accuracy scores
# we get by changing the n_neighbors argument of the k-NN model
k_range = range(1, 30)
k_values = []

# Create a for loop that tests the model accuracy and calculates
# the 10-fold cross valdiation scores by changing the neighbors argument
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=27, test_size=0.3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy score of knn for " + str(k) + " neighbors: " + str(accuracy_score(y_test, y_pred)))
    k_values.append(accuracy_score(y_test, y_pred))
    cv = cross_val_score(knn, features, target, cv=10)
    print("Mean value of 10-fold cross validation: " + str(cv.mean()))
    print(" ")
# Print the list of accuracy scores we got by changing the neighbors
print(k_values)

# Plot a line graph that visualizes how the accuracy of our model
# changed by playing with the n_neighbors argument
sns.set_context("notebook")
sns.set_style("darkgrid")
ax = plt.subplots(figsize=(8, 5))
sns.lineplot(k_range, k_values)
plt.title("KNN accuracy plot")
plt.xlabel("Neighbors")
plt.ylabel("Accuracy score")
plt.show()

# Create a confusion matrix of the entire Bundesliga dataset
matrix_bundesliga = confusion_matrix(y_test, y_pred)
ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrix_bundesliga, annot=True, xticklabels=["draw", "win", "lose"], yticklabels=["draw", "win", "lose"], fmt="g")
plt.title("Confusion matrix of predicted and actual outcomes of Bundesliga games")
plt.xlabel("Actual results")
plt.ylabel("Predicted results")
plt.show()

# Draw countplots of predicted and actual results for the Bundesliga dataset
sns.set_context("talk")
sns.set_style("darkgrid")
ax = plt.subplots(figsize=(8, 5))
sns.countplot(y_pred, order=["w", "d", "l"])
plt.title("Predicted results")
plt.xlabel("Results")
plt.ylabel("Game counts")
plt.ylim(0, 550)
plt.show()
ax = plt.subplots(figsize=(8, 5))
sns.countplot(y_test, order=["w", "d", "l"])
plt.title("Actual results")
plt.xlabel("Results")
plt.ylabel("Game counts")
plt.ylim(0, 550)
plt.show()

# Create the k-NN model declaring 26 as the n_neighbors argument
# since that provided the best accuracy score on the above graph
knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(X_train, y_train)
leipzig_pred = knn.predict(features_leipzig)
# Create a dataframe of predicted Leipzig game results so that
# we can use .value_counts() while comparing the predictions.
leipzig_pred_df = pd.DataFrame(leipzig_pred)
leipzig_pred_df.columns = ["result"]

# Plot the confusion matrix of Leipzig game predictions against
# actual results of Leipzig games
matrix_leipzig = confusion_matrix(target_leipzig, leipzig_pred)
ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrix_leipzig, annot=True, xticklabels=["draw", "win", "lose"], yticklabels=["draw", "win", "lose"])
plt.title("Confusion matrix of predicted and actual outcomes of Leipzig games")
plt.xlabel("Actual results")
plt.ylabel("Predicted results")
plt.show()

# Draw the countplot of predicted Leipzig game results
sns.set_context("talk")
sns.set_style("darkgrid")
ax = plt.subplots(figsize=(8, 5))
sns.countplot(leipzig_pred, order=["w", "d", "l"])
plt.title("Predicted results")
plt.xlabel("Results")
plt.ylabel("Game counts")
plt.ylim(0, 75)
plt.show()
print("Predicted counts of Leipzig games' results:")
print((leipzig_pred_df["result"].value_counts()))
# Draw the countplot of actual Leipzig game results
ax = plt.subplots(figsize=(8, 5))
sns.countplot(target_leipzig, order=["w", "d", "l"])
plt.title("Actual results")
plt.xlabel("Results")
plt.ylabel("Game counts")
plt.ylim(0, 75)
plt.show()
print("Actual counts of Leipzig games' results:")
print(target_leipzig.value_counts())
