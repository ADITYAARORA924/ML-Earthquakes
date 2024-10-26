# ML-Earthquakes
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the diffusion-based model (PDE solver)
class DiffusionModel(nn.Module):
    def _init_(self):
        super(DiffusionModel, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  # Predict seismic wave values

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load seismic data (for simplicity, we generate random data)
def load_data():
    # Replace this with real seismic data loading
    num_samples = 1000
    X = np.random.rand(num_samples, 1, 16, 16)  # Simulated seismic data
    y = np.random.rand(num_samples, 1)  # Simulated seismic wave outputs
    return X, y

# Train the model
def train_model(model, X_train, y_train, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Predict seismic waves
def predict(model, X_test):
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(inputs)
    return predictions.numpy()

if _name_ == "_main_":
    # Initialize the model
    model = DiffusionModel()

    # Load training data
    X_train, y_train = load_data()

    # Train the model
    train_model(model, X_train, y_train)

    # Make predictions (use test data or future seismic data)
    X_test, _ = load_data()  # Replace with actual test data
    predictions = predict(model, X_test)
    print("Predictions:", predictions)
