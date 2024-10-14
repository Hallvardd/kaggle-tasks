import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.optim import Adam
from matplotlib import pyplot as plt
import cv2


def make_autosplit_from_path(data_path: str, split_ratio: float):
    files = list(Path(data_path).glob('*'))
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]
    return train_files, val_files


def anno_to_mask(segmentation: list, image_size:tuple, category_id:int):
    mask = np.zeros(image_size)
    for seg in segmentation:
        seg = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [seg], category_id)
    return mask


class FloodDataset(Dataset):
    def __init__(self, images):
        self.images = images
        # transform the images
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # transform the masks
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = Path(str(image_path).replace('Image', 'Mask').replace('.jpg', '.png'))
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask)

        return image, mask

    def __len__(self):
        return len(self.images)


class MicroControllerDataset(Dataset):
    def __init__(self, data_path: str, anno_path: str, image_transform, mask_transform):
        self.data_path = data_path
        self.anno_path = anno_path
        self.coco = COCO(anno_path)
        self.data = list(self.coco.imgs.values())
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        image_data = self.data[index]
        image_path = image_data['file_name']
        image_id = image_data['id']
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        mask = None
        for a in anno:
            segmentation = a['segmentation']
            category_id = a['category_id']
            tmp_mask = anno_to_mask(segmentation, (image_data['height'], image_data['width']), category_id)
            if mask is None:
                mask = tmp_mask
            else:
                mask += tmp_mask
        image = Image.open(Path(self.data_path)/ image_path)

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask)

        return image, mask

    def __len__(self):
        return len(self.data)


# define the transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the flood dataset
flood_data_path = '../datasets/flood_segmentation/Image/'
flood_train_files, flood_val_files = make_autosplit_from_path(flood_data_path, 0.8)

flood_train_dataset = FloodDataset(flood_train_files)
flood_train_dataloader = DataLoader(flood_train_dataset, batch_size=4, shuffle=True)

flood_val_dataset = FloodDataset(flood_val_files)
flood_val_dataloader = DataLoader(flood_val_dataset, batch_size=4, shuffle=False)


image, mask = flood_train_dataset[0]
image = image.permute(1, 2, 0)
plt.imshow(image)
plt.show()

# convert back from grayscale tenors to grayscale images
mask = mask.permute(1, 0)
plt.imshow(mask)
plt.show()

# microcontroller dataset
microcontroller_train_data_path = '../datasets/microcontroller-segmentation/train2017'
microcontroller_train_anno_path = '../datasets/microcontroller-segmentation/annotations/instances_train2017.json'

microcontroller_val_data_path = '../datasets/microcontroller-segmentation/val2017'
microcontroller_val_anno_path = '../datasets/microcontroller-segmentation/annotations/instances_val2017.json'

microcontroller_train_dataset = MicroControllerDataset(microcontroller_train_data_path, microcontroller_train_anno_path, image_transform, mask_transform)
microcontroller_train_dataloader = DataLoader(microcontroller_train_dataset, batch_size=4, shuffle=True)

microcontroller_val_dataset = MicroControllerDataset(microcontroller_val_data_path, microcontroller_val_anno_path, image_transform, mask_transform)
microcontroller_val_dataloader = DataLoader(microcontroller_val_dataset, batch_size=4, shuffle=False)


# Define the U-Net++ model for multi-class segmentation
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # Use ResNet34 as the encoder (backbone)
    encoder_weights="imagenet",     # Pre-trained on ImageNet
    in_channels=3,                  # RGB input
    classes=2,                      # 5 classes (4 microcontroller classes + 1 background class)
    activation=None                 # No activation, we'll use CrossEntropyLoss which includes softmax
)


# Define the loss function (Cross Entropy for multi-class classification)
loss_fn = torch.nn.CrossEntropyLoss()


# Define the optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


# Training loop
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device):
    model.train()  # Set the model to training mode
    total_train_loss = 0

    for images, masks in train_dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # [B, num_classes, H, W]

        # Compute loss
        loss = loss_fn(outputs, masks.long())  # Make sure masks are long/int for CrossEntropyLoss

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for images, masks in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_fn(images, masks.long())
            total_val_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += masks.numel()
            correct += (predicted == masks).sum().item()

        val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(val_loss)
    # train loss
    train_loss = total_train_loss / len(train_dataloader)

    # Print statistics for the epoch
    print(f'Epoch {epoch+1}, Loss: {train_loss}, '
          f'Validation Loss: {val_loss}, '
          f'Validation Accuracy: {correct/total}')

    return train_loss


# Example training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(20):  # Train for 10 epochs
    avg_loss = train(model, flood_train_dataloader, flood_val_dataloader, loss_fn, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
