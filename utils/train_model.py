import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import copy

def main():
    # 1. Aggressive Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            # added perspective warp to fix the angled shots issue
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            # THE BREED BIAS DESTROYER (Forces model to look at structure, not color)
            transforms.RandomGrayscale(p=1.0),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            
            # Make sure validation is also grayscale for a fair test!
            transforms.RandomGrayscale(p=1.0),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # IMPORTANT: Updated to read from the new auto-cropped folder
    data_dir = 'dataset_cropped'
    if not os.path.exists(data_dir):
        print(f"Error: Cannot find '{data_dir}'. Make sure you ran auto_cropper.py first!")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Build the Model & PARTIAL FINE-TUNING
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # First, freeze EVERYTHING
    for param in model.parameters():
        param.requires_grad = False

    # Second, UNFREEZE only the final convolutional block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3. Build the Custom Classification Head with DROPOUT
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),        
        nn.ReLU(),                       
        nn.Dropout(0.5),                 
        nn.Linear(128, len(class_names)) 
    )
    # (The new fully connected layers automatically have requires_grad=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # 4. Filter parameters to ONLY send the unfrozen ones to the optimizer
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    # We use 1e-4 because training convolutional layers requires smaller, more careful steps
    optimizer = optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-4)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 25
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting V5 Training (Aggressive Augmentation + Grayscale)...")
    
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
    torch.save(model.state_dict(), 'best_cow_classifier_v5.pth')
    print("Saved as 'best_cow_classifier_v5.pth'")

if __name__ == '__main__':
    main()