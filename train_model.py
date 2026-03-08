import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy

def main():
    # 1. Setup Data Augmentation and Normalization
    # We alter the training data slightly each time so the model learns the cow, not the photo
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load the Dataset
    data_dir = 'dataset_split'
    
    # Check if the folder exists before running
    if not os.path.exists(data_dir):
        print(f"Error: Cannot find '{data_dir}'. Did you run split_dataset.py?")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Classes found: {class_names}")
    print(f"Training images: {dataset_sizes['train']} | Validation images: {dataset_sizes['val']}")

    # 3. Setup Hardware (Use GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")

    # 4. Build the Model (Transfer Learning via ResNet18)
    print("Downloading pre-trained ResNet18 brain...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    # Change the final layer to output only 2 classes (0_Healthy, 1_Sick)
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 5. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. The Training Loop
    num_epochs = 10
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Clear old gradients

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize (only in training)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy we've seen
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Training complete! Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best model weights and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_cow_classifier.pth')
    print("Saved the most accurate model as 'best_cow_classifier.pth'")

if __name__ == '__main__':
    # This prevents errors on Windows when using PyTorch DataLoader
    main()