"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np

# DONE by Benoit : Image Normalization is implemented here but I'm not sure if it's correct.
class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)
        # perform z-score normalization on non-zero voxels to preserve background
        mask_nonzero = img_arr != 0

        # if no non-zero voxels, nothing to normalize
        if not np.any(mask_nonzero):
            warnings.warn('Image appears empty (all zeros). Returning unprocessed image.')
            return image

        # compute mean/std on non-zero voxels
        nonzero_vals = img_arr[mask_nonzero].astype(np.float64)
        mean = nonzero_vals.mean()
        std = nonzero_vals.std()

        if std == 0 or np.isclose(std, 0.0):
            warnings.warn('Standard deviation is zero; returning unprocessed image.')
            return image

        # normalize in-place on a float copy
        img_arr_norm = img_arr.astype(np.float32, copy=True)
        img_arr_norm[mask_nonzero] = (img_arr_norm[mask_nonzero] - mean) / std

        img_out = sitk.GetImageFromArray(img_arr_norm)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask

# DONE by Benoit  : Skull Stripping is implemented here but I'm not sure if it's correct.
class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The skull-stripped image.
        """
        mask = None if params is None else getattr(params, 'img_mask', None)

        # validate mask
        if mask is None:
            raise ValueError('SkullStripping requires a brain mask in params.img_mask.')

        # if geometries differ, resample mask to image geometry (nearest neighbor to preserve labels)
        try:
            if (mask.GetSize() != image.GetSize()
                    or mask.GetOrigin() != image.GetOrigin()
                    or mask.GetSpacing() != image.GetSpacing()
                    or mask.GetDirection() != image.GetDirection()):
                mask = sitk.Resample(mask,
                                     image,
                                     sitk.Transform(),
                                     sitk.sitkNearestNeighbor,
                                     0,
                                     mask.GetPixelID())
        except Exception:
            warnings.warn('Mask resampling failed; continuing with provided mask.')

        # binarize mask: treat any non-zero voxel as brain
        mask_bin = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535, insideValue=1, outsideValue=0)

        # apply mask: set voxels outside the brain to zero
        img_masked = sitk.Mask(image, mask_bin, outsideValue=0)
        img_masked.CopyInformation(image)

        return img_masked

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth

# DONE by Benoit  : Image Resgistration is implemented here but I'm not sure if it's correct.
class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """
        # Apply the provided transformation to map the input image into the atlas space.
        # Validate params first before accessing attributes.
        if params is None:
            raise ValueError('ImageRegistration requires params with atlas and transformation.')

        atlas = getattr(params, 'atlas', None)
        transform = getattr(params, 'transformation', None)
        is_ground_truth = getattr(params, 'is_ground_truth', False)

        if atlas is None or transform is None:
            raise ValueError('ImageRegistration requires both params.atlas and params.transformation.')

        # Choose interpolation and output pixel type
        if is_ground_truth:
            interpolator = sitk.sitkNearestNeighbor
            out_pixel_id = image.GetPixelID()
            default_background = 0
        else:
            interpolator = sitk.sitkLinear
            out_pixel_id = sitk.sitkFloat32
            default_background = 0.0

        # Perform resampling: map input image into atlas space using provided transform
        try:
            resampled = sitk.Resample(image,
                                      atlas,
                                      transform,
                                      interpolator,
                                      default_background,
                                      out_pixel_id)
        except Exception as e:
            warnings.warn(f'Resampling (registration) failed; returning original image. Error: {e}')
            return image

        # Ensure spatial metadata matches atlas (Resample already uses atlas geometry, but ensure consistency)
        try:
            resampled.CopyInformation(atlas)
        except Exception:
            # CopyInformation may fail for some image types; ignore to keep result usable
            pass

        return resampled

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
