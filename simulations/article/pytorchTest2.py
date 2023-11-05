import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

# Define a function to visualize the feature maps
def visualize_feature_maps(model, data_loader, device):
    # Get a single batch from the data loader
    images, _ = next(iter(data_loader))

    # We'll collect the feature maps after each Conv2D layer
    feature_maps = []

    # Now we need to redefine the model such that we get the output of each Conv2D layer
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Extract the three convolutional layers from the original model
            self.conv1 = model.model[0]
            self.conv2 = model.model[2]
            self.conv3 = model.model[4]

        def forward(self, x):
            # Pass through the first convolutional layer and store the feature map
            x1 = self.conv1(x)
            x = nn.ReLU()(x1)
            feature_maps.append(x.detach())

            # Pass through the second convolutional layer and store the feature map
            x2 = self.conv2(x)
            x = nn.ReLU()(x2)
            feature_maps.append(x.detach())

            # Pass through the third convolutional layer and store the feature map
            x3 = self.conv3(x)
            x = nn.ReLU()(x3)
            feature_maps.append(x.detach())

            return x3

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(clf).to(device)

    # Forward pass to get the feature maps
    _ = feature_extractor(images.to(device))

    return feature_maps

# Initialize the data loader
train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
data_loader = DataLoader(train_data, batch_size=1)

# Define the neural network architecture from the provided script
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model
clf = ImageClassifier().to('cpu')

# Use the visualization function to get the feature maps
feature_maps = visualize_feature_maps(clf, data_loader, 'cpu')

# Visualize the feature maps
# We will use matplotlib to plot the feature maps
import matplotlib.pyplot as plt

# Function to plot all feature maps in a grid
def plot_all_feature_maps(feature_maps):
    for layer_num, fmap in enumerate(feature_maps):
        layer_name = f"Layer {layer_num+1}"
        num_feature_maps = fmap.size(1)  # Get the number of channels

        # Create a grid of subplots
        fig, axes = plt.subplots(1, num_feature_maps, figsize=(num_feature_maps * 2, 2))
        fig.suptitle(layer_name)

        # Plot each feature map
        for i in range(num_feature_maps):
            # Remove the axis
            axes[i].axis('off')
            # Extract the data and plot it
            axes[i].imshow(fmap[0, i].cpu().detach().numpy(), cmap='gray')
        
        plt.show()

# Call the function to plot all feature maps
plot_all_feature_maps(feature_maps)
