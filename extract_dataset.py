import nibabel as nib
import os
import glob
import numpy as np

output_train_path = 'processed_data/train'
output_valid_path = 'processed_data/valid'
target_size = (512, 512)

def read_nifti_file(filepath, return_affine=False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" The file at '{filepath}' does not exist.")

    nii_img = nib.load(filepath)
    nii_img = nib.as_closest_canonical(nii_img)
    data = nii_img.get_fdata()

    if return_affine:
        return data, nii_img.affine

    return data

data_root = 'data/LiTS'

vol_candidates = glob.glob(os.path.join(data_root, '**', 'volume-*.nii*'), recursive=True)
seg_candidates = glob.glob(os.path.join(data_root, '**', 'segmentation-*.nii*'), recursive=True)

# 3. Helper function to extract ID from filename
# e.g., 'path/to/volume-15.nii' -> returns 15
def get_id(path):
    filename = os.path.basename(path)
    # Split by '-' to get the number part, then split by '.' to remove extension
    id_str = filename.split('-')[1].split('.')[0]
    return int(id_str)

# 4. Create dictionaries mapping { ID : FilePath }
# This is crucial to handle cases where a volume or segmentation might be missing
vol_dict = {get_id(p): p for p in vol_candidates}
seg_dict = {get_id(p): p for p in seg_candidates}

# 5. Find common IDs (IDs present in BOTH lists)
common_ids = sorted(list(set(vol_dict.keys()) & set(seg_dict.keys())))

vols_path = [vol_dict[i] for i in common_ids]
segs_path = [seg_dict[i] for i in common_ids]

def process_volume(volume_path, mask_path, volume_id, mode='train'):
    img_data = read_nifti_file(volume_path)
    mask_data = read_nifti_file(mask_path)
    total_slices = img_data.shape[2]
    
    # Because background slices account for high proportion of dataset,
    # so we need to limit it to get only 10% of it each volume.
    MAX_BACKGROUND_SLICES = total_slices * 0.1    
    cnt_bg = 0

    for i in range(total_slices):
        slice_img = img_data[:, :, i]
        slice_mask = mask_data[:, :, i]

        # 0 = background, 1 = liver, 2 = tumor
        has_tumor = np.any(slice_mask == 2)
        has_liver = np.any(slice_mask == 1)

        if not has_tumor and not has_liver:
            cnt_bg += 1

        if cnt_bg >= MAX_BACKGROUND_SLICES:
            continue

        # --- SAVE ---
        save_image_name = f"vol_{volume_id}_image_{i}"
        save_mask_name = f"vol_{volume_id}_mask_{i}"

        if mode == 'train':
            np.save(os.path.join(output_train_path, 'images', save_image_name), slice_img)
            np.save(os.path.join(output_train_path, 'masks', save_mask_name), slice_mask)
        elif mode == 'valid':
            np.save(os.path.join(output_valid_path, 'images', save_image_name), slice_img)
            np.save(os.path.join(output_valid_path, 'masks', save_mask_name), slice_mask)

for index, path in enumerate(vols_path):
    if index < 35:
        process_volume(vols_path[index], segs_path[index], index, 'train')
    else:
        process_volume(vols_path[index], segs_path[index], index, 'valid')
