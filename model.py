import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

base_data_path = 'C:/Users/admin/Desktop/AAH/sign-language-recognition-project/data'
train_path = base_data_path + '/train'
test_path = base_data_path + '/test'

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(train_path, transform=data_transforms)
test_data = datasets.ImageFolder(test_path, transform=data_transforms)

NUM_CLASSES = len(train_data.classes)

BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, images_arr.shape[0], figsize=(30, 20))
    axes = axes.flatten()
    for img_tensor, ax in zip(images_arr, axes):
        img_tensor = torch.from_numpy(img_tensor)
        img_tensor = img_tensor * 0.5 + 0.5
        img = np.transpose(img_tensor.numpy(), (1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),      
            nn.ReLU(),
            nn.MaxPool2d(2),          

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),          

            nn.Conv2d(64, 128, 3),    
            nn.ReLU(),
            nn.MaxPool2d(2)         
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            out = self.features(dummy)
            self.flattened_size = out.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SignLanguageCNN(num_classes=NUM_CLASSES)
print("\n--- PyTorch Model Summary ---")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

EARLY_STOP_PATIENCE = 7

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                num_epochs=40, patience=7):
    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        print(f"\nEpoch {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss /= len(test_loader.dataset)
        val_acc = correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model_slr.pth')
            print("  Model improved and saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  Early stopping triggered.")
                break

    return history


history = train_model(
    model, train_loader, test_loader,
    criterion, optimizer, scheduler,
    num_epochs=40, patience=7
)

model.load_state_dict(torch.load('best_model_slr.pth'))
model.eval()

test_inputs, test_labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(test_inputs)
    test_loss = criterion(outputs, test_labels).item()
    _, predicted = torch.max(outputs.data, 1)
    test_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

print("\n--- Evaluation on one test batch ---")
print(f"Loss: {test_loss:.4f}, Accuracy: {test_accuracy*100:.2f}%")

print("\nClass mapping:", train_data.class_to_idx)

word_dict = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
}

predictions_indices = predicted.numpy()
print("\nPredictions:")
for ind in predictions_indices:
    print(word_dict.get(ind, 'Unknown'), end='   ')
print("\n")

actual_indices = test_labels.numpy()
print("Actual:")
for i in actual_indices:
    print(word_dict.get(i, 'Unknown'), end='   ')
print()

