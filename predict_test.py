import numpy as np
import ISR
from ISR.models import RDN, RRDN
from PIL import Image
import os

# Get the directory path of the ISR module
isr_module_path = os.path.dirname(ISR.__file__)
print("Path of ISR module:", isr_module_path)

# Initialize the RRDN model
weights_path = "/tf/workspace/weights/rrdn-C4-D3-G32-G032-T10-x4/2023-08-24_1414/rrdn-C4-D3-G32-G032-T10-x4_best-val_generator_PSNR_Y_epoch098.hdf5"
arch_params = {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10}
model = RRDN(arch_params=arch_params, weights_path=weights_path)

# Specify the path to the input image
input_img_path = "/tf/workspace/data/DIV2K/DIV2K/DIV2K_valid_LR_bicubic/159_DT8.png"

# Load the input image using PIL
input_img = Image.open(input_img_path)

# Convert the input image to a numpy array
input_img_array = np.array(input_img)

# Ensure the image has 3 channels (remove alpha channel)
if input_img_array.shape[2] == 4:
    input_img_array = input_img_array[:, :, :3]

print("Input image shape:", input_img_array.shape)

# Perform super-resolution using the model
sr_img = model.predict(input_img_array)
#sr_img = model.predict(input_img_array, by_patch_of_size=50)

print("Super-resolved image shape:", sr_img.shape)

# Convert the super-resolved numpy array to a PIL image
sr_pil_img = Image.fromarray(sr_img)

# Save the super-resolved image with a prefix in the same folder as the input image
output_path = "/tf/workspace/output_super_resolved" # image-super-resolution/output_super_resolved
output_filename = "predicted_" + os.path.basename(input_img_path)
sr_pil_img.save(os.path.join(output_path, output_filename))

print("Super-resolved image saved:", os.path.join(output_path, output_filename))
