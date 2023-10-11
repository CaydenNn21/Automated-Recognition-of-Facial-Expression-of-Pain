import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score

class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CRNN, self).__init__()
        # CNN layers for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # Input channels: 3 (RGB), Output channels: 64, Kernel size: 3x3
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling with kernel size: 2x2 and stride: 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # Input channels: 64, Output channels: 128, Kernel size: 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # Input channels: 128, Output channels: 256, Kernel size: 3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # RNN layers for sequential processing
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True,
                          batch_first=True)  # Input size: input_size, Hidden size: hidden_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        x = self.cnn(x)  # Pass input through the CNN layers
        x = x.permute(0, 3, 1, 2)  # Reshape to (batch_size, seq_len, channels, height, width)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the spatial dimensions

        outputs, _ = self.rnn(x)  # Pass flattened input through the RNN layers
        x = outputs[:, -1, :]  # Extract the last output of the RNN sequence

        x = self.fc(x)  # Pass the output through the fully connected layer

        return x


# Define the hyperparameters
input_size = 256
hidden_size = 128
num_classes = 10
learning_rate = 0.001
batch_size = 16
num_epochs = 10

# Create the CRNN model
model = CRNN(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example training loop
for epoch in range(num_epochs):
    # Data loading and batching
    # Load the dataset
    dataset = load_Data(...)

    # Preprocess the dataset
    preprocessed_dataset = preprocess_dataset(dataset)

    # Split the dataset
    train_set, val_set, test_set = split_dataset(preprocessed_dataset, train_ratio, val_ratio, test_ratio)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize variables for computing metrics
    true_labels = []
    predicted_labels = []
    for i, (inputs, labels) in enumerate(train_loader):
        # A train_loader python file will be defined once the dataset that used to train the model is confirmed

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record true and predicted labels for computing metrics
        true_labels.extend(labels.tolist())
        predicted_labels.extend(outputs.argmax(dim=1).tolist())

        # Compute training accuracy
    train_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Training Accuracy: {train_accuracy}")

    # Compute and display the confusion matrix
    train_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Training Confusion Matrix:")
    print(train_confusion_matrix)

    # Validation and model evaluation
    model.eval()
    with torch.no_grad():
        # Initialize variables for computing metrics
        true_labels = []
        predicted_labels = []

    for val_batch in val_loader:
        val_inputs, val_labels = val_batch['frames'], val_batch['pain_scores']
        val_outputs = model(val_inputs)
# Final testing
model.eval()
with torch.no_grad():
    # Initialize variables for computing metrics
    true_labels = []
    predicted_labels = []

    for test_batch in test_loader:
        test_inputs, test_labels = test_batch['frames'], test_batch['pain_scores']
        test_outputs = model(test_inputs)

        # Record true and predicted labels for computing metrics
        true_labels.extend(test_labels.tolist())
        predicted_labels.extend(test_outputs.argmax(dim=1).tolist())

    # Compute testing accuracy
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Testing Accuracy: {test_accuracy}")

    # Compute and display the confusion matrix
    test_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Testing Confusion Matrix:")
    print(test_confusion_matrix)
