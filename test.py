import os
import torch
from botorch.test_functions.multi_objective import BraninCurrin
import torch
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from typing import Optional, Tuple
from torch import Tensor
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

tkwargs = {
    "dtype": torch.float32,
    "device": torch.device("cuda" if torch.cuda.is_available() else "mps"),
}


problem = BraninCurrin(negate=True).to(**tkwargs).to(torch.float32)

def get_reduced_cifar10(train_size=8000, test_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the full datasets
    download = True
    if os.path.exists("data"):
      download = False
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform, )
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)

    # Perform stratified sampling on the training set
    train_labels = [label for _, label in full_trainset]
    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    train_idx, _ = next(sss_train.split(train_labels, train_labels))

    # Perform stratified sampling on the test set
    test_labels = [label for _, label in full_testset]
    sss_test = StratifiedShuffleSplit(n_splits=1, train_size=test_size, random_state=42)
    test_idx, _ = next(sss_test.split(test_labels, test_labels))

    # Create subset datasets
    trainset = Subset(full_trainset, train_idx)
    testset = Subset(full_testset, test_idx)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers, num_filters, fc_size):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential()
        in_channels = 3
        for i in range(num_conv_layers):
            self.features.add_module(f'conv{i}', nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            self.features.add_module(f'relu{i}', nn.ReLU())
            self.features.add_module(f'pool{i}', nn.MaxPool2d(2, 2))
            in_channels = num_filters

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * (32 // (2**num_conv_layers))**2, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# @use_named_args(dimensions)
def train_and_evaluate(num_conv_layers, num_filters, fc_size, learning_rate):
    print("gello")
    num_conv_layers = int(num_conv_layers)
    num_filters = int(num_filters)**2
    fc_size = int(fc_size)**2

    # Load CIFAR-10 dataset
    trainloader, testloader = get_reduced_cifar10(train_size=8000, test_size=2000)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)
    model = SimpleCNN(num_conv_layers, num_filters, fc_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # print(model)
    # summary(model, (3, 32, 32))

    # Train the model
    print("training start: ")
    for epoch in range(50):  # Reduced number of epochs for quicker optimization
        epoch_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"{epoch_loss}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    model_size = sum(p.numel() for p in model.parameters())

    return torch.tensor([accuracy, model_size])

class ImageClassifierProblem(MultiObjectiveTestProblem):
    r"""Two objective problem for image classification.
    Objectives:
    1. Number of parameters (to be minimized)
    2. Model accuracy (to be maximized)

    Hyperparameters:
    1. Learning rate (log scale)
    2. Number of filters
    3. Number of layers
    """
    dim = 3
    num_objectives = 2
    _bounds = [(1e-4, 1e-1), (3, 6), (2.0, 5)]
    _ref_point = [1e7, 0.5]  # 10M parameters, 50% accuracy

    def __init__(
        self,
        noise_std: Optional[Tensor] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _compute_objectives(X) -> Tensor:
        num_params = []
        accuracy = []

        for lr, num_filters, num_layers in X:
          # Dummy calculations - replace these with your actual model training and evaluation
          acc, params = train_and_evaluate(num_layers, num_filters, 5, lr)
          accuracy.append(acc)
          num_params.append(params)

        accuracy = torch.tensor(accuracy)
        num_params = torch.tensor(num_params)

        return torch.stack([-num_params, accuracy], dim=-1)

    def evaluate_true(self, X):
        # print("hello")
        return self._compute_objectives(X)



if __name__ == "__main__":
    problem = ImageClassifierProblem(negate=False)  # negate=True because BoTorch maximizes by default

    X = torch.tensor([[0.01, 5, 2], [0.006, 4, 3]])
    print("Proceeding to training and evaluation")
    train_and_evaluate(2, 3, 3, 0.01)
    # problem(X)
    