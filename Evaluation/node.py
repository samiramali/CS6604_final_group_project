import pickle

#https://stellargraph.readthedocs.io/en/stable/demos/node-classification/node2vec-node-classification.html

# embeddings = pickle.load(open('embeddings.pkl', 'rb'))

# print(embeddings.shape)

# targets = pickle.load(open('targets.pkl', 'rb'))

# print(targets[35])
# print(targets.shape)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Specify the path to your text file
file_path = "new_embedding_real_cross_dependency/embedding_layer_0_real_cross.txt"
layer_0 = np.loadtxt(file_path, delimiter=',')

file_path = "embeddings/embedding_layer_1.txt"
layer_1 = np.loadtxt(file_path, delimiter=',')

file_path = "embeddings/embedding_layer_2.txt"
layer_2 = np.loadtxt(file_path, delimiter=',')

layer_0 = np.concatenate((layer_0, layer_1, layer_2), axis=0)

# Load the text file into a NumPy array, preserving its shape
# Assuming the file contains numeric layer_0 with consistent column

print("Shape of the layer 0:", layer_0.shape)
print("Shape of the layer 1:", layer_1.shape)
print("Shape of the layer 2:", layer_2.shape)

k = 5  # You can adjust this value based on your requirements

# Perform k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(layer_0)

# Get the cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Print results
print("Cluster labels:", labels)
print("Cluster centers:", centers)

#Visualize the clustering (for 2D layer_0)
plt.scatter(layer_0[:, 0], layer_0[:, 1], c=labels, cmap='viridis', s=30)
#plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
# plt.title("K-Means Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(layer_0, labels, test_size=0.3, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Visualize classification boundaries (only if input layer_0 is 2D)
import matplotlib.pyplot as plt


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap='viridis', values_format='d')
plt.title("Confusion Matrix")
plt.show()


