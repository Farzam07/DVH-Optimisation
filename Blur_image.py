# ==================================I M P O R T S=============================================================
import os
import numpy as np
import subprocess as sub
import shutil
import argparse
import math
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd
import platform
import SimpleITK as sitk
import time
import pydicom
from scipy import signal
from scipy.interpolate import interp1d
from datetime import datetime

# =============================================P A R S E R=============================================================
parser = argparse.ArgumentParser(description='Blur an inputted image with a varying range of Gaussian filters')
parser.add_argument('-input_image','--input_image',type=str,help='path to the image you wish to blur')
parser.add_argument('--image_name',type=str,help='Name for the saved blurred images: Name_FWHMxmm')
parser.add_argument('-min', '--min', type=float, help='Minimum FWHM value of the gaussian filter')
parser.add_argument('-max', '--max', type=float, help='Maximum FWHM value of the gaussian filter')
parser.add_argument('-step', '--step', type=float, help='The step between each FWHM value')
parser.add_argument('-o', '--output', type=str, help='Path to output folder')

args = parser.parse_args()

if not os.path.exists(os.path.join(args.output)):
    os.makedirs(os.path.join(args.output))

for fwhm_mm in np.arange(args.min, args.max+args.step, args.step):  # FWHM evaluation range from 2 mm to 20 mm

    fwhm_mm = round(fwhm_mm, 1)
    var = ((fwhm_mm) * (fwhm_mm)) / (8 * math.log(2))

    list_args_clitkBlurImage = ['clitkBlurImage',
                                '-i', args.input_image,
                                '-o', os.path.join(args.output,f'{args.image_name}_Blurred_FWHM' + str(fwhm_mm) + '.nii.gz'),
                                '--variance', str(var),
                                '-v']
    sub.run(list_args_clitkBlurImage)
