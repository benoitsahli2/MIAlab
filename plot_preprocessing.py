"""Script to visualize MRI image before normalization, after normalization, and after skull stripping.

This script loads a T1-weighted MRI image from the training set and displays it
in its original form, after normalization, and after skull stripping.
"""
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

# Import preprocessing filters from mialab
from mialab.filtering.preprocessing import ImageNormalization, SkullStripping, SkullStrippingParameters

# Define paths
subject_id = '100307'
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
image_path = os.path.join(data_dir, 'train', subject_id, 'T1native.nii.gz')
mask_path = os.path.join(data_dir, 'train', subject_id, 'Brainmasknative.nii.gz')

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Brain mask not found: {mask_path}")

# Load image and brain mask
image = sitk.ReadImage(image_path)
brain_mask = sitk.ReadImage(mask_path)
img_arr = sitk.GetArrayFromImage(image)

# Apply normalization
normalizer = ImageNormalization()
image_norm = normalizer.execute(image)
img_arr_norm = sitk.GetArrayFromImage(image_norm)

# Apply skull stripping to normalized image
skull_stripper = SkullStripping()
params = SkullStrippingParameters(brain_mask)
image_stripped = skull_stripper.execute(image_norm, params)
img_arr_stripped = sitk.GetArrayFromImage(image_stripped)

# Plot middle slice for all three stages
slice_idx = img_arr.shape[0] // 2
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot original image
im1 = axes[0].imshow(img_arr[slice_idx], cmap='gray')
axes[0].set_title('Original MRI')
plt.colorbar(im1, ax=axes[0])

# Plot normalized image
im2 = axes[1].imshow(img_arr_norm[slice_idx], cmap='gray')
axes[1].set_title('After Normalization')
plt.colorbar(im2, ax=axes[1])

# Plot skull-stripped image
im3 = axes[2].imshow(img_arr_stripped[slice_idx], cmap='gray')
axes[2].set_title('After Skull Stripping')
plt.colorbar(im3, ax=axes[2])

# Add statistics as text
axes[0].text(0.02, 0.98, f'Mean: {np.mean(img_arr):.2f}\nStd: {np.std(img_arr):.2f}', 
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1].text(0.02, 0.98, f'Mean: {np.mean(img_arr_norm):.2f}\nStd: {np.std(img_arr_norm):.2f}',
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[2].text(0.02, 0.98, f'Mean: {np.mean(img_arr_stripped[img_arr_stripped != 0]):.2f}\nStd: {np.std(img_arr_stripped[img_arr_stripped != 0]):.2f}',
             transform=axes[2].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
