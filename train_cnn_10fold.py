import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# === Configuration ===
class CFG:
    root_dir = "Local"
    metadata_csv = "metadata.csv"
    batch_size = 32
    num_epochs = 3
    lr = 1e-4
    n_folds = 10
    img_size = (224, 224)
    seed = 42

# === Dataset Definition ===
class SpectrogramImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(self.dataframe['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['file_name'])
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[self.dataframe.iloc[idx]['label']]
        if self.transform:
            image = self.transform(image)
        return image, label

# === Main Cross-Validation Training Function ===
def train_kfold(df):  # Accept filtered DataFrame as parameter
    num_classes = df['label'].nunique()

    transform = transforms.Compose([
        transforms.Resize(CFG.img_size),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n=== Fold {fold+1} ===")

        train_dataset = SpectrogramImageDataset(df.iloc[train_idx], CFG.root_dir, transform=transform)
        val_dataset = SpectrogramImageDataset(df.iloc[val_idx], CFG.root_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

        for epoch in range(CFG.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Fold {fold+1} Validation Accuracy: {acc:.4f}")
        results.append(acc)

    print("\n=== Cross-Validation Summary ===")
    print(f"Mean Accuracy: {sum(results)/len(results):.4f}")
    return results

# === Entry Point ===
if __name__ == "__main__":
    df = pd.read_csv(CFG.metadata_csv)

    # Filter out rare species (e.g., fewer than 10 samples)
    min_required = 10
    vc = df['label'].value_counts()
    df = df[df['label'].isin(vc[vc >= min_required].index)].reset_index(drop=True)

    # Optional: Print removed classes
    print("Removed rare species:\n", vc[vc < min_required])

    # Pass filtered DataFrame to training function
    train_kfold(df)
