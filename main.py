import argparse
from ISR.models import RRDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer

"""
Train an ISR model using PSNR-only or GAN training mode.

Usage:
    python main.py --mode psnr_only
    python main.py --mode psnr_only --weights /path/to/weights
    python main.py --mode gan
    python main.py --mode gan --weights /path/to/weights
    python main.py

Args:
    --mode (str): Training mode. (Default is gan)
    --weights (str): Path to pre-trained generator weights for transfer training. (Default is None)

Troubleshooting:
    If you are having trouble loading your own weights or the pre-trained weights (AttributeError: 'str' object has no attribute 'decode'), try:

    pip install 'h5py==2.10.0' --force-reinstall

ISR Repository:
    https://github.com/idealo/image-super-resolution/tree/master#pre-trained-networks
"""

# Initialize argparse
parser = argparse.ArgumentParser(description="ISR Training Main")

# Add mode and weights arguments
parser.add_argument('--mode', type=str, choices=["psnr_only", "gan"], default="gan", help="Training mode")
parser.add_argument('--weights', type=str, default=None, help="Path to pre-trained generator weights for transfer training")

# Parse the command-line arguments
args = parser.parse_args()

# Determine the loss weights based on the mode
if args.mode == "psnr_only":
    loss_weights = {
        'generator': 1.0,
        'feature_extractor': 0.0,
        'discriminator': 0.00
    }
    training_mode_str = "PSNR-only training"
else:
    loss_weights = {
        'generator': 0.0,
        'feature_extractor': 0.0833,
        'discriminator': 0.01
    }
    training_mode_str = "GAN training"

# Determine weights_generator based on --weights argument
if args.weights and args.weights.lower() != "none":
    weights_generator = args.weights
else:
    weights_generator = None

## Setup Models
'''
Import the models from the ISR package and create
    - generator RRDN super scaling network
    - discriminator network for GANs training
    - VGG19 feature extractor to train with a perceptual loss function

Carefully select:
    'x': this is the upscaling factor (2 by default)
    'layers_to_extract': these are the layers from the VGG19 that will be used in the perceptual loss (leave the default if you're not familiar with it)
    'lr_patch_size': this is the size of the patches that will be extracted from the LR images and fed to the ISR network during training time
'''

lr_train_patch_size = 40 # [40]
layers_to_extract = [5, 9]
scale = 4 # [2]
hr_train_patch_size = lr_train_patch_size * scale

#rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':32, 'G0':32, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

print("Models setup completed.")

## Setup Trainer
'''
Parse the arguments in the call
Initialize hyper-parameters
Initialize the trainer

'''

log_dirs = {'logs': './logs', 'weights': './weights'}
learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}
flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}


# Initialize Trainer
trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='./data/DIV2K/DIV2K/DIV2K_train_LR_bicubic',
    hr_train_dir='./data/DIV2K/DIV2K/DIV2K_train_HR',
    lr_valid_dir='./data/DIV2K/DIV2K/DIV2K_valid_LR_bicubic',
    hr_valid_dir='./data/DIV2K/DIV2K/DIV2K_valid_HR',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='div2k',
    log_dirs=log_dirs,
    weights_generator=weights_generator,
    weights_discriminator=None,
    n_validation=40,
)

print("Trainer setup completed.")

# Choose epoch number, steps, and batch size and start training
trainer.train(
    epochs=50 if args.mode == "psnr_only" else 300, # Change as you see fit. Suggested values are 50+ for PSNR only training.
    steps_per_epoch=20, # [20]
    batch_size=4, # [4]
    monitored_metrics={'val_generator_PSNR_Y': 'max'}
)

print(f"{training_mode_str} completed.")