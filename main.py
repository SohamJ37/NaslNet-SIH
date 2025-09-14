import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

from PIL import Image

train_folder = "C:\\Users\\Jain\\Desktop\\SIH Project\\CATTLE BREED DATASET"
valid_folder = "C:\\Users\\Jain\\Desktop\\SIH Project\\CATTLE BREED DATASET"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = TrainingDataset(train_folder, transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TrainingDataset(valid_folder, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for images, labels in train_dataloader:
    break


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    

model = SimpleClassifier(num_classes=41)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_dataloader, val_dataloader):
    num_epoch = 5
    train_losses, val_losses = [], []
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.00
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss/len(train_dataloader.dataset)
        train_losses.append(train_loss)


        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        val_loss = running_loss/len(val_dataloader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {train_loss}, Validation Loss: {val_loss}")




class_names = train_dataset.classes
infer_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_image(path, model, device):
    model.eval()
    img = Image.open(path).convert("RGB")
    x = infer_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

        top_prob, top_idx = torch.max(probs, dim=0)
        return probs.cpu()
    
def predict_multiple(paths, model, device, class_names):
    total_probs = torch.zeros(len(class_names))

    for path in paths:
        probs = predict_image(path ,model, device)
        total_probs += probs

    top_idx  = torch.argmax(total_probs).item()
    confidence = total_probs[top_idx].item()

    return class_names[top_idx], confidence, total_probs
    

# train(model, train_dataloader, val_dataloader)
# torch.save(model.state_dict(), "breed.pth")
