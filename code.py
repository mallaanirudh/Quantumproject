import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns  
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Generate a complex dataset
np.random.seed(42)  # For reproducibility
n_samples = 1000
time = np.random.randint(1, 100, size=n_samples)  # Random time values
amount = np.random.randint(100, 1000, size=n_samples)  # Random amount values
label = np.random.randint(0, 2, size=n_samples)  # Random binary labels (0 or 1)

# Combine into a dataset
data = np.column_stack((time, amount, label))

# Separate the features and labels
X = data[:, :2]  # First two columns are features (time and amount)
y = data[:, 2]   # Third column is the label (0 or 1)

# Step 2: Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 4: Amplitude Encoding
def amplitude_encoding(data):
    norm_data = data / np.linalg.norm(data)
    qml.templates.AmplitudeEmbedding(norm_data, wires=[0, 1], pad_with=0, normalize=True)

# Step 5: Build Quantum Fourier Transform
def qft(wires):
    qml.Hadamard(wires=wires[0])
    for i in range(1, len(wires)):
        for j in range(i):
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(np.pi / 2**(i - j), wires=wires[i])
    for i in range(len(wires) // 2):
        qml.SWAP(wires=[wires[i], wires[len(wires) - i - 1]])

# Step 6: Build Hybrid QNN Model
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    amplitude_encoding(inputs)
    qml.RY(weights[0], wires=0)  # Example of a parameterized rotation
    qft(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))  # Return expectation values

def build_qnn_model(input_shape):
    weight_shapes = {"weights": (1,)}  # Shape for weights

    model = keras.Sequential()
    model.add(qml.qnn.KerasLayer(quantum_circuit, 
                                  output_dim=2,  # Output dimensions match the QNode output
                                  input_shape=input_shape, 
                                  weight_shapes=weight_shapes))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Define input shape and build model
input_shape = (2,)  # Two features
model = build_qnn_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model (changed epochs to 5)
model.fit(X_train, y_train, epochs=5, batch_size=4, validation_split=0.2)

# Step 8: Make Predictions
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Step 9: Evaluate the Model
print("Classification Report:")
print(classification_report(y_test, predicted_labels))

conf_matrix = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Step 10: Visualizations
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 11: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, predictions)  # Make sure to use the raw predictions for ROC
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()
