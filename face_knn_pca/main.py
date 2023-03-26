import statistics
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, cosine


# Custom KNN Classifier Class
class KNNClassifier:
    # Initializing the classifier
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    # Training data and calculating co-variance   
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        if self.distance_metric == 'mahalanobis':
            self.covariance = np.cov(X_train.T)
    # Predicting results
    def predict(self, X_test):
        y_pred = []
        # Loop over each row in test set
        for x_test in X_test:
            distances = []
            # Loop over each row in train set and find distance
            for i, x_train in enumerate(self.X_train):
                if self.distance_metric == 'euclidean':
                    dist = np.sqrt(np.sum((x_test - x_train) ** 2))
                elif self.distance_metric == 'mahalanobis':
                    dist = mahalanobis(x_test, x_train, self.covariance)
                elif self.distance_metric == 'cosine':
                    dist = cosine(x_test, x_train)
                else:
                    raise ValueError('Invalid distance metric')
                distances.append((i, dist))
            # Sort distances so that shortest ones are in front
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:self.k]
            neighbor_labels = [self.y_train[i] for i, _ in neighbors]
            # Most frequent label in k nearest neighbours
            y_pred.append(max(set(neighbor_labels), key=neighbor_labels.count))
        return y_pred


# Hyper parameters
dm = 'euclidean'
k = 5
n_subjects = 10
n_total_per_subject = 170
n_train_per_subject = 150
n_val_per_subject = n_total_per_subject - n_train_per_subject
n_repeats = 5
pca_components = 75
print(f'Distance Metric: {dm}, K: {k}, NC: {n_subjects}, n_pca: {pca_components}')
print(f'Total({n_total_per_subject}) = Train({n_train_per_subject}) + Val({n_val_per_subject})')


# Load and preprocess data
data = pd.read_csv('fea.csv', header=None)
data = data.to_numpy()

# Calculate norm and normalize X on it
magnitudes = np.linalg.norm(data, axis=1)
X = data / magnitudes[:, np.newaxis]

# Generating labels 0 to 9 for 170 samples
y = np.repeat(np.arange(n_subjects), n_total_per_subject)

# Reduce features to pca_components
if pca_components:
    pca = PCA(n_components=pca_components)
    X = pca.fit_transform(X)

# # Calculate covariance and correlation
# cov = np.cov(X.T)
# corr = np.corrcoef(X.T)
#
# # Visualize the covariance matrix as a heatmap
# plt.imshow(cov, cmap='jet')
# plt.title('Covariance Matrix Heatmap')
# plt.colorbar()
# plt.show()
#
# # Check if the data is uncorrelated
# if np.allclose(corr, np.identity(corr.shape[0])):
#     print('The data is uncorrelated')
# else:
#     print('The data is correlated')


# Initialize arrays to store the training and testing indices for each repeat
train_indices_all = np.zeros((n_repeats, n_subjects * n_train_per_subject), dtype=int)
test_indices_all = np.zeros((n_repeats, n_subjects * n_val_per_subject), dtype=int)

# Loop over the repeats
for r in range(n_repeats):
    # Initialize arrays to store the training and testing indices for this repeat
    train_indices = np.zeros(n_subjects * n_train_per_subject, dtype=int)
    test_indices = np.zeros(n_subjects * n_val_per_subject, dtype=int)

    # Loop over the subjects
    for s in range(n_subjects):
        # Randomly select the indices for training and testing samples for this subject
        all_indices = np.arange(s * n_total_per_subject, (s + 1) * n_total_per_subject)
        train_indices_sub = np.random.choice(all_indices, size=n_train_per_subject, replace=False)
        test_indices_sub = np.setdiff1d(all_indices, train_indices_sub)

        # Add the indices for this subject to the overall training and testing indices
        train_indices[s * n_train_per_subject:(s + 1) * n_train_per_subject] = train_indices_sub
        test_indices[s * n_val_per_subject:(s + 1) * n_val_per_subject] = test_indices_sub

    # Shuffle the indices for this repeat and add them to the overall indices
    perm = np.random.permutation(len(train_indices))
    train_indices_all[r] = train_indices[perm]
    perm = np.random.permutation(len(test_indices))
    test_indices_all[r] = test_indices[perm]


accuracies = []
times = []
# Loop over the repeats
for split in range(n_repeats):
    # Training and Test data for this repeat
    X_train = X[train_indices_all[split]]
    y_train = y[train_indices_all[split]]
    X_test = X[test_indices_all[split]]
    y_test = y[test_indices_all[split]]

    # Initialize and fit the KNN classifier
    knn = KNNClassifier(k, distance_metric=dm)
    knn.fit(X_train, y_train)

    # Make predictions on the test set and record computational time
    t = time.time()
    y_pred = knn.predict(X_test)
    time_taken = time.time() - t
    times.append(time_taken)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Generating report
avg_accuracy = sum(accuracies) / len(accuracies)
std_dev = statistics.stdev(accuracies)
avg_comp_time = sum(times) / len(times)
print("Avg Accuracy: {:.3f}".format(avg_accuracy))
print("Standard Deviation: {:.3f}".format(std_dev))
print("Avg Comp Time: {:.3f}".format(avg_comp_time), "s")
