import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import copy

def main():
    # 1. UPGRADED: Aggressive Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Keep more of the cow in frame
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Tilt the images randomly
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Vary the lighting
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

    data_dir = 'dataset_split'
    
    if not os.path.exists(data_dir):
        print(f"Error: Cannot find '{data_dir}'.")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Training images: {dataset_sizes['train']} | Validation images: {dataset_sizes['val']}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build the Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # 2. UPGRADED: Lower Learning Rate + Weight Decay (prevents memorization)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 3. UPGRADED: Learning Rate Scheduler (Drops LR by 10% every 7 epochs)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. UPGRADED: Train for 25 Epochs
    num_epochs = 25
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler only after the training phase
            if phase == 'train':
                step_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(">>> New Best Model Saved!")

        print()

    print(f'Training complete! Best Validation Accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_cow_classifier_v2.pth')
    print("Saved as 'best_cow_classifier_v2.pth'")

if __name__ == '__main__':
    main()