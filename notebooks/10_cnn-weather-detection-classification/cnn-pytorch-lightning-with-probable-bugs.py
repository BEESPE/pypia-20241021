# code pytorch + pytorch lightning, généré automatiquement à partir du code tensorflow
# probablement non fonctionnel, juste pour avoir des idées de la syntaxe 

import os
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Path to images and label list
IMG_FOLDER = os.getenv("IMG_FOLDER")
LABELS_LIST = ["cloudy", "rain", "shine", "sunrise"]

# Helper functions


def get_image_label(img_filename: str, possible_labels=None, label_not_found_value="unknown") -> str:
    """Extract label from the image file name."""
    if possible_labels is None:
        possible_labels = LABELS_LIST
    for label in possible_labels:
        if label in img_filename:
            return label
    return label_not_found_value


def create_data_from_img_files(path: str = IMG_FOLDER) -> pd.DataFrame:
    """Create DataFrame with image paths and labels."""
    images_list = glob(f"{path}*/*.jp*")
    df_out = pd.DataFrame(data={"image_path": images_list})
    df_out["label_name"] = df_out["image_path"].apply(get_image_label)
    return df_out

# Custom Dataset for loading images


class ImageDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        self.transform = transform
        self.label_mapping = {label: i for i, label in enumerate(LABELS_LIST)}

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df.iloc[idx]["image_path"]
        label = self.img_df.iloc[idx]["label_name"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_mapping[label]
        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create DataFrame and split dataset
img_df = create_data_from_img_files()
X_train, X_test = train_test_split(
    img_df, stratify=img_df['label_name'], test_size=0.2, random_state=0)
X_train, X_val = train_test_split(
    X_train, stratify=X_train['label_name'], test_size=0.2, random_state=0)

# Create datasets and dataloaders
train_dataset = ImageDataset(X_train, transform=transform)
val_dataset = ImageDataset(X_val, transform=transform)
test_dataset = ImageDataset(X_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Lightning Model Definition


class VGG16Lightning(pl.LightningModule):
    def __init__(self, num_classes=4):
        super(VGG16Lightning, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze pretrained layers
        self.vgg.classifier[6] = nn.Linear(
            4096, num_classes)  # Modify the last layer
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.vgg(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.vgg.classifier[6].parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(outputs, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss


# Model creation
model = VGG16Lightning(num_classes=4)

# Callbacks for checkpoint and early stopping
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

# Trainer
trainer = pl.Trainer(
    max_epochs=10,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Training
trainer.fit(model, train_loader, val_loader)

# Testing
trainer.test(model, test_loader)

# Confusion matrix and classification report
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Matrice de confusion et rapport de classification
conf_mat = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", classification_report(
    y_true, y_pred, target_names=LABELS_LIST))
