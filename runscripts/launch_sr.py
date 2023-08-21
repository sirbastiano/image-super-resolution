import numpy as np
from PIL import Image
from ISR.models import RDN
import matplotlib.pyplot as plt

img = Image.open('1_DT1x8.png')
lr_img = np.array(img)

rdn = RDN(weights='psnr-small')
model = rdn
sr_img = model.predict(lr_img, by_patch_of_size=50)

# save super resolved image:
sr_img = Image.fromarray(sr_img)
sr_img.save('1_DT1x8_SR.png')
