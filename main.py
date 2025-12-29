import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from tqdm import tqdm

from LiverTumorModel import LiverTumorModel

MODEL_PATH = 'models/best_liver_tumor_model_v45.pth'

VOLUME_FILE_PATH = 'data/LiTS/volume_pt5/volume-46.nii'

# Add path if you have ground true masks
MASK_FILE_PATH = 'data/LiTS/segmentations/segmentation-46.nii'

ROT_K = 1 # rotates 90 degrees. Change to 2 or 3 if needed.
HU_RANGE = (-100, 200) # focuses on Liver/Tumor contrast.

def read_nifti_file(filepath, return_affine=False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" The file at '{filepath}' does not exist.")

    nii_img = nib.load(filepath)
    nii_img = nib.as_closest_canonical(nii_img)
    data = nii_img.get_fdata()

    if return_affine:
        return data, nii_img.affine

    return data


def generate_prediction_volume(model, volume_data):
    """
    Runs the model on every slice of the volume and stacks the results.
    Returns:
        pred_volume: 3D numpy array of masks
        slice_probs: List of max tumor probabilities for each slice
    """
    rows, cols, depth = volume_data.shape

    # Initialize empty 3D mask
    pred_volume = np.zeros((rows, cols, depth), dtype=np.uint8)
    slice_probs = []  # Store probability for each slice

    print(f"Starting prediction on volume shape: {volume_data.shape}...")

    for i in tqdm(range(depth), desc="Predicting slices"):
        # 1. Get the current slice
        img_slice = volume_data[:, :, i]

        # 2. Predict
        # model.predict automatically resizes input to 512x512 and returns 512x512
        pred_mask_slice, prob = model.predict(img_slice)
        slice_probs.append(prob)

        # 3. Resize back if necessary
        # If the original image wasn't 512x512, we must resize the mask back
        # to match the original dimensions so it fits in the viewer.
        if pred_mask_slice.shape[0] != rows or pred_mask_slice.shape[1] != cols:
            pred_mask_slice = cv2.resize(
                pred_mask_slice.astype(float),  # Ensure valid type for resize
                (cols, rows),  # cv2 uses (width, height)
                interpolation=cv2.INTER_NEAREST  # NEAREST to keep classes 0, 1, 2
            )

        # 4. Store in 3D volume
        pred_volume[:, :, i] = pred_mask_slice.astype(np.uint8)

    print("Prediction complete.")
    return pred_volume, slice_probs


class NiiSliceViewer:
    def __init__(self, volume, original_mask=None, predicted_mask=None, slice_probs=None, figsize=(15, 6),
                 hu_range=(-100, 250), rot_k=1):

        # 1. Apply HU Clipping
        self.volume = np.clip(volume, hu_range[0], hu_range[1])

        self.orig_mask = original_mask
        self.pred_mask = predicted_mask
        self.slice_probs = slice_probs
        self.show_ground_truth = True  # Default state for toggle

        # 2. Apply Rotation
        if rot_k > 0:
            print(f"Rotating images by 90 degrees * {rot_k}...")
            self.volume = np.rot90(self.volume, k=rot_k, axes=(0, 1))

            if self.orig_mask is not None:
                self.orig_mask = np.rot90(self.orig_mask, k=rot_k, axes=(0, 1))

            if self.pred_mask is not None:
                self.pred_mask = np.rot90(self.pred_mask, k=rot_k, axes=(0, 1))

        if self.volume.ndim != 3:
            raise ValueError("Input volume must be 3D (Height, Width, Depth)")

        self.rows, self.cols, self.depth = self.volume.shape
        self.ind = self.depth // 2

        # 3. Determine Layout (2 or 3 plots)
        self.has_gt = (self.orig_mask is not None)
        num_plots = 3 if self.has_gt else 2

        self.fig, self.ax = plt.subplots(1, num_plots, figsize=figsize)
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.85)

        self.vmin = hu_range[0]
        self.vmax = hu_range[1]

        self.im_objs = []
        self.mask_objs = []

        # Define Titles and Axes mapping
        if self.has_gt:
            # Layout: [Original, Ground Truth, Prediction]
            self.titles = [
                "Original Slice",
                "Ground Truth (Press 't' to toggle)",
                "Prediction"
            ]
            axes_list = self.ax  # self.ax is a list of 3 axes
        else:
            # Layout: [Original, Prediction]
            self.titles = [
                "Original Slice",
                "Prediction"
            ]
            axes_list = self.ax  # self.ax is a list of 2 axes

        # Initialize Plots
        for i, ax in enumerate(axes_list):
            ax.set_title(self.titles[i], fontsize=10)
            ax.axis('off')

            # Base Image
            img = ax.imshow(self.volume[:, :, self.ind], cmap='gray', vmin=self.vmin, vmax=self.vmax)
            self.im_objs.append(img)
            self.mask_objs.append(None)  # Placeholder for overlay

        # Initial Draw
        self.update_overlays()
        self.update_titles()

        # Slider config
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Slice', 0, self.depth - 1, valinit=self.ind, valstep=1)
        self.slider.on_changed(self.update_slider)

        # Event Listeners
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        status_msg = "Slices: {}. Scroll or use Slider.".format(self.depth)
        if self.has_gt:
            status_msg += " Press 't' to toggle Ground Truth."
        print(f"Viewer initialized. {status_msg}")

        plt.show()

    def get_colored_overlay(self, mask_slice):
        """Converts a 2D mask slice (0, 1, 2) into an RGBA image."""
        overlay = np.zeros((self.rows, self.cols, 4))
        # Liver (1) -> Green
        overlay[mask_slice == 1] = [0, 1, 0, 0.3]
        # Tumor (2) -> Red
        overlay[mask_slice == 2] = [1, 0, 0, 0.6]
        return overlay

    def update_overlays(self):
        # Determine which index the Prediction plot is at
        # If GT exists, Pred is at index 2. If no GT, Pred is at index 1.
        pred_idx = 2 if self.has_gt else 1

        # 1. Update Ground Truth (Only if exists)
        if self.has_gt:
            # Handle visibility toggle
            if self.show_ground_truth:
                mask_slice = self.orig_mask[:, :, self.ind]
                overlay = self.get_colored_overlay(mask_slice)

                if self.mask_objs[1] is None:
                    self.mask_objs[1] = self.ax[1].imshow(overlay)
                else:
                    self.mask_objs[1].set_visible(True)
                    self.mask_objs[1].set_data(overlay)
            else:
                # If toggled off, hide the image object if it exists
                if self.mask_objs[1] is not None:
                    self.mask_objs[1].set_visible(False)

        # 2. Update Prediction
        if self.pred_mask is not None:
            pred_slice = self.pred_mask[:, :, self.ind]
            overlay = self.get_colored_overlay(pred_slice)

            if self.mask_objs[pred_idx] is None:
                self.mask_objs[pred_idx] = self.ax[pred_idx].imshow(overlay)
            else:
                self.mask_objs[pred_idx].set_data(overlay)

    def update_titles(self):
        """Updates the prediction title with the current slice probability."""
        if self.slice_probs is not None:
            current_prob = self.slice_probs[self.ind]
            prob_text = f"\nMax Tumor Prob: {current_prob * 100:.2f}%"

            # Identify which plot holds the prediction
            pred_idx = 2 if self.has_gt else 1
            base_title = self.titles[pred_idx]

            self.ax[pred_idx].set_title(base_title + prob_text, fontsize=10,
                                        color='blue' if current_prob > 0.5 else 'black')

    def update(self):
        # Update base grayscale images
        for img in self.im_objs:
            img.set_data(self.volume[:, :, self.ind])

        # Update colored masks
        self.update_overlays()

        # Update titles
        self.update_titles()

        self.fig.canvas.draw_idle()

    def update_slider(self, val):
        self.ind = int(self.slider.val)
        self.update()

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.depth
        elif event.button == 'down':
            self.ind = (self.ind - 1) % self.depth
        self.slider.set_val(self.ind)

    def on_key_press(self, event):
        # Only toggle if we actually have ground truth
        if event.key == 't' and self.has_gt:
            self.show_ground_truth = not self.show_ground_truth
            self.update()


if __name__ == "__main__":
    # 1. Initialize Model
    # Ensure you are on the correct device (cuda if available)
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    model = LiverTumorModel(MODEL_PATH, device=device)

    # 2. Load Data
    print("Loading NIfTI files...")
    vol = read_nifti_file(VOLUME_FILE_PATH)
    gt_mask = None
    if os.path.exists(MASK_FILE_PATH):
        gt_mask = read_nifti_file(MASK_FILE_PATH)
    else:
        print(f"Mask file not found at {MASK_FILE_PATH}. Viewer will skip Ground Truth column.")

    # 3. Generate Prediction
    # This will run the model on all slices and return a 3D numpy array
    pred_mask, probs = generate_prediction_volume(model, vol)

    # 4. Launch Viewer with Rotation and Clipping
    app = NiiSliceViewer(vol, gt_mask, pred_mask, slice_probs=probs, rot_k=ROT_K, hu_range=HU_RANGE)
