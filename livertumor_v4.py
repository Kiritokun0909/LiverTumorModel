import shutil
import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Config dataset path from runtime
TRAIN_IMG_DIR = '/content/dataset/train/images/'
TRAIN_MASK_DIR = '/content/dataset/train/masks/'
TEST_IMG_DIR = '/content/dataset/valid/images/'
TEST_MASK_DIR = '/content/dataset/valid/masks/'
ROOT_DIR = 'content/dataset'

RESUME = False  # False if train from start, else will load last checkpoint and continue training
CHECKPOINT_PATH = '/content/drive/MyDrive/Colab/model_v4/check_point/liver_checkpoint.pth'
BEST_MODEL_PATH = '/content/drive/MyDrive/best_liver_tumor_model.pth'

class TransformFunction:
    def __init__(self, rotate_prob=0.5, noise_prob=0.5, max_angle=15, noise_sigma=0.1):
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.max_angle = max_angle
        self.noise_sigma = noise_sigma

    def __call__(self, image, mask):
        # --- 1. Random Rotation (Applied to both Image and Mask) ---
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = TF.rotate(image, angle)
            # Use interpolation=nearest for masks to avoid introducing new classes (e.g. 0.5)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # --- 2. Gaussian Noise (Applied to Image only) ---
        if random.random() < self.noise_prob:
            noise = torch.randn_like(image) * self.noise_sigma
            image = image + noise

        return image, mask

class LiverTumorDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        self.image_dir = os.path.join(root_dir, phase, 'images')
        self.mask_dir = os.path.join(root_dir, phase, 'masks')
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, 'vol_*_image_*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, 'vol_*_mark_*.npy')))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        mask_filename = filename.replace('_image_', '_mask_')
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # --- 1. Convert image to float32 datatype ---
        try:
            image = np.load(image_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        # --- 2. Preprocessing (Clipping HU) ---
        image = np.clip(image, -100, 250)

        # --- 3. Dimension Handling (H, W -> C, H, W) ---
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        # --- 4. Convert to Tensor ---
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # --- 5. Apply Augmentations ---
        # Only apply augmentations if a transform is provided AND we are in training
        if self.transform and self.phase == 'train':
            image, mask = self.transform(image, mask)

        return image, mask

# Load dataset into Datasets & DataLoaders
# Initialize your augmentation module
augmenter = TransformFunction(
    rotate_prob=0.5,
    max_angle=20,     # Rotate +/- 20 degrees
    noise_prob=0.5,
    noise_sigma=0.3  # Noise intensity
)

train_dataset = LiverTumorDataset(
    root_dir=ROOT_DIR,
    phase='train',
    transform=augmenter
)
valid_dataset = LiverTumorDataset(
    root_dir=ROOT_DIR,
    phase='valid',
    transform=None
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)


# --- DEVICE CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- Initialize UNet++ architecture model with resnet backbone ---
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=1,                  
    classes=3,                      
    activation=None
).to(device)

# --- LOSS FUNCTION & OPTIMIZER ---
criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
class_weights = torch.tensor([0.1, 1.0, 5.0]).to(device)

criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


def calculate_metrics(pred_logits, target_mask):
    pred_labels = torch.argmax(pred_logits, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(pred_labels, target_mask, mode='multiclass', num_classes=3)

    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")  # F1_score is also dice_loss
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

    return iou, f1_score, accuracy


def save_checkpoint(state, filename):
    print("=> Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    print(f"=> Loading checkpoint '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']

    print(f"=> Loaded checkpoint (epoch {start_epoch}, best_score {best_score:.4f})")
    return start_epoch, best_score
    
    

SAVE_PATH = '/content/drive/MyDrive/Colab/model_v5/best_liver_tumor_model.pth'

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)

        masks = masks.squeeze(1).long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion_dice(logits, masks) + criterion_ce(logits, masks)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        iou, dice, acc = calculate_metrics(logits, masks)

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        running_acc += acc

        pbar.set_postfix({'Loss': loss.item(), 'IoU': iou, 'Dice': dice, 'Accuracy': acc})

    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_acc = running_acc / len(loader)
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)

            logits = model(images)
            loss = criterion_dice(logits, masks) + criterion_ce(logits, masks)

            iou, dice, acc = calculate_metrics(logits, masks)

            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            running_acc += acc

            pbar.set_postfix({'Val Loss': loss.item(), 'Val IoU': iou, 'Accuracy': acc})

    return running_loss / len(loader), running_iou / len(loader), running_dice / len(loader), running_acc/len(loader)

# --- Training configuration ---
EPOCHS = 10
best_iou = 0.0

if RESUME and os.path.exists(CHECKPOINT_PATH):
    start_epoch, best_iou = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
else:
    start_epoch = 0
    print("=> Starting fresh training (No checkpoint loaded)")

print(f"Training from epoch {start_epoch + 1} to {EPOCHS}...")


print("Start Training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_iou, train_dice, train_acc = train_one_epoch(model, train_dataloader, optimizer, device)
    val_loss, val_iou, val_dice, val_acc = validate(model, valid_dataloader, device)

    print(f"Epoch {epoch+1} Summary:")
    print(f"Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f} | Acc: {train_acc:.4f}")
    print(f"Valid - Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_score': best_iou,
    }
    save_checkpoint(checkpoint, CHECKPOINT_PATH)

    curr_val = val_iou * 0.5 + val_dice * 0.5
    if curr_val > best_iou:
        print(f"Validation IoU improved ({best_iou:.4f} -> {val_iou:.4f}). Saving model...")
        best_iou = curr_val
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")

print("Training Complete.")

import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2

class LiverPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=3,
            activation=None
        )

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_numpy):
        if image_numpy.shape[0] != 512 or image_numpy.shape[1] != 512:
            image_numpy = cv2.resize(image_numpy, (512, 512))

        image = np.clip(image_numpy, -100, 250)

        min_hu, max_hu = -100.0, 250.0
        # image = (image - min_hu) / (max_hu - min_hu)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        return image_tensor

    def predict(self, image_numpy):
        input_tensor = self.preprocess(image_numpy).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        return predicted_mask

print("\n--- DEMO PREDICTION ---")
test_img_path = '/content/dataset/train/images/vol_0_image_46.npy'
test_img = np.load(test_img_path)
predictor = LiverPredictor(model_path=SAVE_PATH, device=device)

result_mask = predictor.predict(test_img)

print(f"Input Image Shape: {test_img.shape}")
print(f"Predicted Mask Shape: {result_mask.shape}")
print(f"Unique values in prediction: {np.unique(result_mask)}")

test_img = np.clip(test_img, -100, 200)
test_img = np.rot90(test_img)
result_mask = np.rot90(result_mask)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Predicted Mask (0:Background, 1:Liver, 2:Tumor)")
plt.imshow(result_mask, cmap='jet', alpha=0.7)
plt.show()