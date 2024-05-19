import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

# Define the directory where your data is located
data_dir = 'dataset'


def data_loader(data_dir):
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into training and testing subsets
    train_size = int(0.8 * len(dataset))  # 80% training, 20% testing
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing data
    batch_size = 32  # Specify the batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optionally, you can get the class names from the dataset
    class_names = dataset.classes

    return train_loader, test_loader, class_names




class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1_input_size = 32 * 56 * 56  # Adjusted input size based on output size
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Apply convolutional and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the output before feeding it to the fully connected layers
        x = x.view(-1, self.fc1_input_size)
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fit_model(train_loader, model):

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed


    # Number of training epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Print accuracy after each epoch
        print('Epoch [%d] Training Accuracy: %.2f %%' % (epoch + 1, 100 * correct / total))

    print('Finished Training')
    return model


def model_evaluation(class_names, test_loader, model):
    # Evaluate the model on the test dataset
    correct = 0
    total = 0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
            ground_truths.extend(labels.tolist())

    # Compute accuracy
    accuracy = 100 * correct / total
    print('Accuracy on the test dataset: %.2f %%' % accuracy)

    # Compute other metrics (e.g., precision, recall, F1-score) using scikit-learn
    
    print('Classification Report:')
    print(classification_report(ground_truths, predictions, target_names=class_names))
    return model

def save_model(model):
    # Save the trained model
    torch.save(model, 'trained_model.pth')
    return 'Model saved successfully.'

if __name__ == '__main__':
    # Load the dataset
    train_data_dir = 'dataset'
    train_loader, test_loader, class_names = data_loader(train_data_dir)
    
    # initiate the model architecture
    cnn_model = CNNModel(len(class_names))

    # fit the model with training data 
    start_model = fit_model(train_loader, cnn_model)

    # evaluate the model
    model_output_summary = model_evaluation(class_names, test_loader, start_model)

    saved_model = save_model(start_model)
    print(saved_model)
    

