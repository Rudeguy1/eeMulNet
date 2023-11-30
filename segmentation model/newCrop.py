
import json
import nibabel as nib
import numpy as np
from nilearn.image.image import _crop_img_to as crop_img_to
import os
import re
import SimpleITK as sitk
from tqdm import tqdm
import pandas



def get_slice_index(data, rtol=1e-8):
    """Obtain the position coordinates of the cropping space"""
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)

    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]
    return slices


def have_back(image):
    """Use True and False to mark the background area, which is False"""
    background_value=0
    tolerance=0.00001
    is_foreground = np.logical_or(image.get_fdata() < (background_value - tolerance),
                                  image.get_fdata()> (background_value + tolerance))

    foreground = np.zeros(is_foreground.shape, dtype=np.float64)
    foreground[is_foreground] = 1
    return foreground

def crop_image_with_label(image_path, label_path, outputpath, extract_maxcroparea=True):
    # Read the Nii image and label, extract the image area corresponding to the label, crop the 0 value,
    # and save the extracted image coordinates for possible subsequent operations
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path, sitk.sitkInt32)
    connected_component = sitk.ConnectedComponent(label)
    labeled_object = sitk.RelabelComponent(connected_component, 1)
    # Extract all connected domains
    extracted_image = sitk.Mask(image, labeled_object > 0)
    # Extract the maximum area of the label image
    if extract_maxcroparea:
        # Convert the extracted image into a Numpy array
        extracted_array = sitk.GetArrayFromImage(extracted_image)

        # Calculate the maximum rectangular area with grayscale values that are not 0
        nonzero_indices = np.nonzero(extracted_array)
        min_x = np.min(nonzero_indices[0])
        max_x = np.max(nonzero_indices[0])
        min_y = np.min(nonzero_indices[1])
        max_y = np.max(nonzero_indices[1])
        min_z = np.min(nonzero_indices[2])
        max_z = np.max(nonzero_indices[2])
        min_max_xyz = [min_x, max_x, min_y, max_y, min_z, max_z]
        # Crop Image
        cropped_array = extracted_array[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

        cropped_image = sitk.GetImageFromArray(cropped_array)

        #Save cropped image
        sitk.WriteImage(cropped_image, outputpath)
    else:
        sitk.WriteImage(extracted_image, outputpath)
        min_max_xyz = 0
    return min_max_xyz


def crop2_image_with_label(image_path, label_path, outputpath, extract_maxcroparea=True):
    # Read the Nii image and label, extract the image area corresponding to the label, crop the 0 value,
    # and save the extracted image coordinates for possible subsequent operations
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path, sitk.sitkInt32)
    origin = label.GetOrigin()
    space = label.GetSpacing()
    derection = label.GetDirection()

    # Extracting Irregular Shapes Using Binary Morphology Operations
    connected_component = sitk.ConnectedComponent(label)
    labeled_object = sitk.RelabelComponent(connected_component, 1)
    # Extract all connected domains
    extracted_image = sitk.Mask(image, labeled_object > 0)
    # Extract the maximum area of the label image
    if extract_maxcroparea:
        extracted_array = sitk.GetArrayFromImage(extracted_image)

        # Calculate the maximum rectangular area with grayscale values that are not 0
        nonzero_indices = np.nonzero(extracted_array)
        min_x = np.min(nonzero_indices[0])
        max_x = np.max(nonzero_indices[0])
        min_y = np.min(nonzero_indices[1])
        max_y = np.max(nonzero_indices[1])
        min_z = np.min(nonzero_indices[2])
        max_z = np.max(nonzero_indices[2])
        min_max_xyz = [min_x, max_x, min_y, max_y, min_z, max_z]
        # Crop Image
        cropped_array = extracted_array[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

        cropped_image = sitk.GetImageFromArray(cropped_array)
        # --------------------------------------------------------------------------
        # Set a new direction matrix from RAI to RPS
        cropped_image.SetOrigin(origin)
        cropped_image.SetSpacing(space)
        cropped_image.SetDirection(derection)
        # ------------------------------------------------------------------------
        #Save cropped image
        sitk.WriteImage(cropped_image, outputpath)
    else:
        sitk.WriteImage(extracted_image, outputpath)
        min_max_xyz = 0
    return min_max_xyz



def obtain_path(image_file, label_file):
    image_paths = [os.path.join(image_file, i) for i in os.listdir(image_file)]
    label_path = [os.path.join(label_file, j) for j in os.listdir(label_file)]
    return image_paths, label_path

def check_correspond_imgLabel(image_paths, label_paths):
    # Check if label sorting and image sorting are consistent
    for image_path, label_path in zip(image_paths, label_paths):
        imgpatient_id = image_path.split('\\')[-1].split('.')[0]
        labelpatient_id = label_path.split('\\')[-1].split('.')[0]
        if imgpatient_id != labelpatient_id:
            assert False, f"label and image not match {image_path},{label_path}"


image_paths, label_paths = obtain_path(image_file=r'',
                                       label_file=r'')
check_correspond_imgLabel(image_paths, label_paths)
out_file = r''
coordinate_data = {}
print(f'is working')
for image_path, label_path in tqdm(zip(image_paths, label_paths)):

    patient_id = image_path.split('\\')[-1]
    output_path = f'{out_file}\\crop_{patient_id}'
    min_max_xyz = crop2_image_with_label(image_path, label_path, output_path, extract_maxcroparea=True)
    if min_max_xyz != 0:
        min_max_xyz = list(map(int, min_max_xyz))
        coordinate_data[image_path] = min_max_xyz
json_data = json.dumps(coordinate_data, indent=4)
with open(f'{out_file}\coordinate_data.json', "w") as file :
    file.write(json_data)
