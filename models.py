import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3), #1
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3), #2
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3), #3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(20, 20, kernel_size=3), #4
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3), #5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6480, 4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        return self.model(x)

    def train_model(self, trainloader, test_loader, loss_fn, n_epochs, optimizer):
        self.to(self.device)
        for epoch in range(n_epochs):
            self.train()
            running_loss = 0.0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}')
            self.test(test_loader)

    def test(self, test_loader):
        self.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        return accuracy

class CNN_cifar(nn.Module):
    def __init__(self):
        super(CNN_cifar, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device {self.device}")

        self.model = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5), #1
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3), #2
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3), #3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(20, 20, kernel_size=3), #4
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3), #5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8000,4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = x.view(x.size(0), 3, 32, 32)
        return self.model(x)

    def train_model(self, trainloader, test_loader, loss_fn, n_epochs, optimizer):
        self.to(self.device)
        for epoch in range(n_epochs):
            self.train()
            running_loss = 0.0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}')
            self.test(test_loader)

    def test(self, test_loader):
        self.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        return accuracy
