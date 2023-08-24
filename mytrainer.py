import ISR
from ISR.models import RRDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer

# Setup Models
'''
Import the models from the ISR package and create

a RRDN super scaling network
a discriminator network for GANs training
a VGG19 feature extractor to train with a perceptual loss function
Carefully select

'x': this is the upscaling factor (2 by default)
'layers_to_extract': these are the layers from the VGG19 that will be used in the perceptual loss (leave the default if you're not familiar with it)
'lr_patch_size': this is the size of the patches that will be extracted from the LR images and fed to the ISR network during training time
Play around with the other architecture parameters
'''

lr_train_patch_size = 40 #[40]
layers_to_extract = [5, 9]
scale = 4 # [2]
hr_train_patch_size = lr_train_patch_size * scale

# rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':32, 'G0':32, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

print("Models setup completed.")

# Setup Trainer
'''
The Trainer object will combine the networks, manage your training data and keep you up-to-date with the training progress through Tensorboard and the command line.

Here we do not use the pixel-wise MSE but only the perceptual loss by specifying the respective weights in loss_weights
'''

loss_weights = {
  'generator': 0.0,
  'feature_extractor': 0.0833,
  'discriminator': 0.01
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
} 

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='./data/DIV2K/DIV2K/DIV2K_train_LR_bicubic',  # Change to the path of your LR training data
    hr_train_dir='./data/DIV2K/DIV2K/DIV2K_train_HR',          # Change to the path of your HR training data
    lr_valid_dir='./data/DIV2K/DIV2K/DIV2K_valid_LR_bicubic',  # Change to the path of your LR test data
    hr_valid_dir='./data/DIV2K/DIV2K/DIV2K_valid_HR',          # Change to the path of your HR test data
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='div2k',
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
)

print("Trainer setup completed.")

# Choose epoch number, steps and batch size and start training

trainer.train(
    epochs=80, # [20]
    steps_per_epoch=500, # [20]
    batch_size=4, # [4]
    monitored_metrics={'val_generator_PSNR_Y': 'max'}
)

print("Training completed.")