from webbrowser import get
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def get_links():
    file_path = "layer_1_adjacency.csv"
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    # Remove the first column
    processed_data = data[:, 1:]

    print(processed_data.shape)

    # Count the occurrences of the value 1 in the array
    count_ones = np.sum(processed_data == 1)

    # Print the result
    print(f"The array contains {count_ones} occurrence(s) of the value 1.")

    row_indices, col_indices = np.indices(processed_data.shape)

    # Flatten the indices and the matrix for easier processing
    row_indices = row_indices.flatten()
    col_indices = col_indices.flatten()
    values = processed_data.flatten()

    # Separate the pairs based on whether they have a link or not
    linked_pairs = np.array(list(zip(row_indices[values == 1], col_indices[values == 1])))
    non_linked_pairs = np.array(list(zip(row_indices[values == 0], col_indices[values == 0])))

    return linked_pairs, non_linked_pairs

# Example node embeddings (1162 nodes, 2-dimensional embeddings)
# Specify the path to your text file
file_path = "new_embedding_real_cross_dependency/embedding_layer_0_real_cross.txt"
embeddings = np.loadtxt(file_path, delimiter=',')

# Generate sample edges for demonstration
node_count = embeddings.shape[0]

print(node_count)
# positive_edges = np.random.randint(0, node_count, size=(2000, 2))  # Example positive edges
# negative_edges = np.random.randint(0, node_count, size=(1000, 2))  # Example negative edges

# print(positive_edges.shape)

positive_edges, negative_edges = get_links()

positive_labels = [1] * len(positive_edges)
negative_labels = [0] * len(negative_edges)

print(positive_edges.shape)
print(negative_edges.shape)


# Combine positive and negative samples
all_edges = np.vstack((positive_edges, negative_edges))
all_labels = np.array(positive_labels + negative_labels)

# Feature construction: Combine node embeddings for each edge
def construct_features(edge):
    u, v = edge
    emb_u = embeddings[u]
    emb_v = embeddings[v]
    # Use concatenation, element-wise multiplication, and absolute difference as features
    return np.concatenate([emb_u, emb_v, emb_u * emb_v, np.abs(emb_u - emb_v)])

features = np.array([construct_features(edge) for edge in all_edges])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, all_labels, test_size=0.3, random_state=42)

# Train a binary classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

# Plot ROC-AUC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.grid()
plt.show()

# Predict link for a specific pair
node_a, node_b = 2, 10  # Example node pair
link_features = construct_features((node_a, node_b))
predicted_label = clf.predict([link_features])[0]
print(f"Link between {node_a} and {node_b}: {'Exists' if predicted_label == 1 else 'Does not exist'}")
