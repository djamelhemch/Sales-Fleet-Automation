import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,  accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

# Define the weights for each criterion
weights = {
    'payment_history': 0.2,
    'lifetime_value': 0.2,
    'demand': 0.3,
    'order_frequency': 0.2,
    'distance': 0.2
}

# Load the data from the file
data = pd.read_json('./outputs/json/modified_training_data3.json')

# Extract relevant features for clustering
features = []
for route in data['routes']:
    for customer in route['assigned_customers']:
        feature = [
            customer['payment_history'],
            customer['lifetime_value'],
            customer['demand'],
            customer['order_frequency'],
            customer['distance']
        ]
        weighted_feature = [weight * value for weight, value in zip(weights.values(), feature)]
        features.append(weighted_feature)
features = np.array(features)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Perform customer clustering using K-means
num_clusters = 4  # A, B, C, D classes
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(scaled_features)
cluster_labels = kmeans.predict(scaled_features)

# Define the neural network model for classifying customers
def create_model(units=32, rate=0.2):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=5, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate))  # Add dropout layer with specified rate
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate))  # Add dropout layer with specified rate
    model.add(Dense(num_clusters, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the neural network model with desired units
best_units = 128
best_rate = 0.2
best_model = create_model(units=best_units, rate=best_rate)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, tf.keras.utils.to_categorical(cluster_labels, num_clusters), test_size=0.2, random_state=42)

# Train the best model
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Evaluate the best model on the test data
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Generate predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels back to original format
y_test_labels = np.argmax(y_test, axis=1)

# Create a classification report
target_names = ['Class A', 'Class B', 'Class C', 'Class D']

y_test_binarized = label_binarize(y_test_labels, classes=np.arange(num_clusters))
y_pred_binarized = label_binarize(y_pred_labels, classes=np.arange(num_clusters))
# Calculate the accuracy
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_labels)
# Calculate the precision, recall, and F1-score for each class
precision = precision_score(np.argmax(y_test, axis=1), y_pred_labels, average=None)
recall = recall_score(np.argmax(y_test, axis=1), y_pred_labels, average=None)
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_labels, average=None)

# Print the metrics for each class
for i, target_name in enumerate(target_names):
    print(f'{target_name}:')
    print(f'Precision: {precision[i]:.4f}')
    print(f'Recall: {recall[i]:.4f}')
    print(f'F1-score: {f1[i]:.4f}')
    print()
# Calculate the false positive rate and true positive rate for ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_clusters):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.plot(fpr[i], tpr[i], label=f'Class {target_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Save the trained model
#best_model.save('best_customer_clustering_model.h5')
