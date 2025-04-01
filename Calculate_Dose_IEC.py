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

# =============================================P A R S E R=====================================================
parser = argparse.ArgumentParser(description='Generate and compare reconstructions using a DVH-based approach')

parser.add_argument('--label_image_folder', type=str, help='Path towards folder containing the labelled image containing the ROIs')
parser.add_argument('--PET_Image_folder', type=str,
                    help='Path towards the folder containing Static PET images (Dicom folders or standard image file)')

parser.add_argument('-dc', '--dicom_flag', help='Include this option if the static PET is in DICOM format',
                    action='store_true')
parser.add_argument('-AT', '--affine_transform_flag',
                    help='Include this option to transform CT size Label to PET size Label before NEMA analysis',
                    action='store_true')

parser.add_argument('--radionuc',type=str,help='Radionucleide (options: Y90)')

parser.add_argument('-k', '--input_path_DVK_file', help='path to the dose kernel file')
parser.add_argument('-k_stat', '--input_path_DVK_stat_file', help='path to the Kernel simulation stat file in txt format')

parser.add_argument('-ref_dose','--input_path_ref_dose_file',type=str,help='path to the reference dose image (ex: GATE dosemap)')
parser.add_argument('-ref_dose_mask','--ref_dose_mask_file',type=str,help='path towards the mask/labelled image for the reference dose image')
parser.add_argument('-ref_dose_stat','--ref_dose_stat_file',type=str,help='path towards the Reference Dose simulation stat file in txt format')
parser.add_argument('-auto_decay','--auto_decay',help='Include this option to automatically determine theoretical activity from Dicom info',action='store_true')
parser.add_argument('-Th_Act_file','--theoretical_act_file',type=str,help='path to the csv file containing the theoretical activities in MBq (Include if auto_decay is off)')
parser.add_argument('-o', '--output', type=str, help='Path to output folder')

args = parser.parse_args()

# ==================================F U N C T I O N S==================================================================

def invDistributionFunction(x,y):
    '''
    Inverse a function based on given x and y values.
    :param x: X values of the distribution
    :param y: Y values of the distribution
    :return: The x and y values of the inversed function
    '''
    #x = vector
    x_new = []
    y_new = []
    i = 0
    while i < (len(x)-1):
        val0 = y(x[i]).item()
        val1 = y(x[i+1]).item()
        if val0 > val1:
            x_new.append(y(x[i]).item())
            y_new.append(x[i])
            i = i + 1
        else:
            while val0 == val1 and i < (len(x)-2): #len(x)-3:
                i = i + 1 #len(x)-2
                val0 = y(x[i]).item()
                val1 = y(x[i+1]).item()
            if i == (len(x)-2):
                x_new.append(y(x[i+1]).item())
                y_new.append(x[i+1])
                i = i + 1
            else:
                x_new.append(y(x[i]).item())
                y_new.append(x[i])
                i = i + 1
    x_new.reverse()
    y_new.reverse()
    return x_new, y_new

def dvhRMSE(x,invDVH, invDVHRef):
    '''
    Calculate Root mean square error between two DVHs
    :param x: X array for relative volume from 0 - 100 with N points
    :param invDVH: Inverse of the image based DVH (LDM or DVK)
    :param invDVHRef: Inverse of the reference MC DVH
    :return:
    '''
    rmse = 0
    sum = 0
    for i in range(len(x)-1):
        #Compute the area under the curve - let's use the rectangle method
        rmse = rmse + (x[i+1]-x[i])*(invDVH(x[i]).item() - invDVHRef(x[i]).item())**2
        sum = sum + (x[i+1]-x[i])
    rmse = rmse/sum
    rmse = np.sqrt(rmse)
    return rmse

def instantiate_object():
    '''
    Instantiate new image object into a list
    :return: image_object_list
    '''
    image_object_list = []

    for image_file in os.listdir(args.PET_Image_folder):
        file_format_im = os.path.splitext(image_file)[-1]
        tag_im = image_file.split('_')[-1].rstrip(file_format_im)  # The characterstic tag
        # Step1: Check if multiple label images exist (we will need to match each label image to the corresponding pet image)
        if len(os.listdir(args.label_image_folder)) > 1:
            for label_file in os.listdir(args.label_image_folder): # Loop over the label files
                file_format_label=os.path.splitext(label_file)[-1]
                tag_label=label_file.split('_')[-1].rstrip(file_format_label) #The characterstic tag
                if tag_label == tag_im: #Check if the same tag is the image_file name
                    image_object_list.append(PetImage(image_file,label_file)) #Instantiate a new object

        else: # If the label folder has only one image, it'll be used for all the pet images
            image_object_list.append(PetImage(image_file,os.listdir(args.label_image_folder)[0]))


    return image_object_list


def plot_rmse(df_rmse_LDM,df_rmse_DVK):
    '''
    :param df_rmse: Dataframe containing the rmse values
    :param method: The dosimetric method (LDM or DVK)
    :return:
    '''
    # Set the bar width
    bar_width = 0.35

    # Set the positions of bars on X-axis
    r1 = np.arange(len(df_rmse_LDM['Name']))
    r2 = [x + bar_width for x in r1]

    # Plot the bars for each column except for the Name column
    columns_to_plot = [col for col in df_rmse_LDM.columns if col != 'Name']

    for column in columns_to_plot:
        bars_ldm =plt.bar(r1, df_rmse_LDM[column], color='b', width=bar_width, edgecolor='grey', label='LDM')
        bars_dvk= plt.bar(r2, df_rmse_DVK[column], color='r', width=bar_width, edgecolor='grey', label='DVK')

        # Add labels and title
        plt.xlabel('Recon Name', fontweight='bold')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.xticks([r + bar_width / 2 for r in range(len(df_rmse_LDM['Name']))], df_rmse_LDM['Name'], rotation=45,ha='right')
        plt.ylabel(f'{column}', fontweight='bold')
        plt.title(f'{column}')
        plt.legend()

        # # Add Y values above the bars
        # for  bar_ldm,bar_dvk in zip(bars_ldm,bars_dvk):
        #     yval_ldm=bar_ldm.get_height()
        #     yval_dvk=bar_dvk.get_height()
        #     plt.text(bar_ldm.get_x() + bar_ldm.get_width() / 2, yval_ldm, round(yval_ldm, 2), ha='center', va='bottom')
        #     plt.text(bar_dvk.get_x() + bar_dvk.get_width() / 2, yval_dvk, round(yval_dvk, 2), ha='center', va='bottom')

        # plt.show()
        plt.tight_layout()
        output_file_name = f'Plot_{column}_{os.path.basename(os.path.abspath(args.PET_Image_folder))}.pdf'
        output_folder_path=os.path.join(args.output, 'RMSE',)
        plt.savefig(os.path.join(output_folder_path,output_file_name))
        plt.close()


# ==================================C L A S S==================================================================

# Create a class for the Image analysis. Each analysed image will be an object of this class
class PetImage():
    # Class object attributes (attributes in common for all objects of the class)
    dicom_flag= args.dicom_flag
    affine_transform_flag=args.affine_transform_flag
    label_image_folder = args.label_image_folder
    pet_image_folder = args.PET_Image_folder
    radionuc_half_life_sec = {'Y90': 230549.76,'F18': 6583.68,'Ga68': 4069.8}
    radionuc_mean_energy_MeV={'Y90':0.935} #MeV

    def __init__(self,pet_image_file,label_image_file):
        self.pet_image_file=os.path.join(self.pet_image_folder,pet_image_file) # PET Image File path
        self.label_image_file=os.path.join(self.label_image_folder,label_image_file) # Label file path
        self.radionuc=args.radionuc #Radionucleide
        self.acq_tag=os.path.split(self.pet_image_file)[-1].split('_')[-1].rstrip('.mha') #Acquisition tag name extracted from image name

        #Initialise attributes that will later take the pet image and label image as numpy arrays
        self.pet_image_arr=None
        self.label_image_arr=None
        self.pet_cum_act_arr=None

        #Initialise dose maps arrays and paths
        self.dose_map_LDM_arr= None
        self.LDM_dosemap_file= None
        self.dose_map_DVK_arr= None
        self.DVK_dosemap_file = None
        self.REF_dosemap_file=None

        #Initialise list that will contain the path to DVHs
        self.LDM_DVH_path =[]
        self.DVK_DVH_path = []
        self.GATE_DVH_path = []

        #Initialise list that will contain the GATE mean dose for each structure
        self.GATE_mean_dose=[]
        #Intialise theoretical activity
        self.activity_MBq=None

        #Intialise nRMSE and RMSE dictionaries
        self.nrmse_LDM = {}
        self.nrmse_DVK = {}

        self.rmse_LDM = {}
        self.rmse_DVK = {}

    def dicom2Image(self,dst): #ADD DST AS A PARAMETER
        '''
         Convert a series of Dicom images into mha format
        :return:
        '''

        #dst = 'C:/Coupes'  # A temporary directory where we will move the DICOM series in order to have a shorter filepath

        if os.path.exists(dst):
            shutil.rmtree(os.path.join(dst))

        shutil.move(self.pet_image_file, dst)

        dirlist = []  # We create an empty list to store the path to all the DICOM files in a series
        [dirlist.append(os.path.abspath(os.path.join(dst, name))) for name in
         os.listdir(dst)]  # We store the path to all the Dicom files (files ending with dcm)
        dirlist = natsorted(dirlist)  # We sort the list by alphabetical/numerical order

        command = ' '.join(dirlist)  # Extract the contents of the dirlist as one string to be used in the clitkDicom2Image function

        output_file_name = os.path.split(self.pet_image_file)[-1] + '.mha'
        output_file_path=os.path.join(args.output, 'mha_images',output_file_name)

        sub.run(f'clitkDicom2Image {command} -o {output_file_path} -t 0.001 -p',shell=True)  # Using a clitk function to perform the Dicom to Image transformation

        shutil.move(dst, self.pet_image_file)

        self.pet_image_folder=os.path.join(os.path.split(os.path.abspath(args.PET_Image_folder))[-2], 'mha_images')

        self.pet_image_file=output_file_path

    def affine_transform(self):
        '''
        Change CT sized label image to PET size and spacing
        '''
        output_file_name=os.path.split(self.label_image_file)[-1].rstrip('.mha') + '_PET.mha'
        out_folderpath=os.path.join(os.path.split(os.path.abspath(args.label_image_folder))[-2], 'PET_sized_label_images')
        out_filepath = os.path.join(out_folderpath, output_file_name)

        sub.run(['clitkAffineTransform', '-i', self.label_image_file, '-l',self.pet_image_file, '--interp', '0', '-o', out_filepath])

        #Modify the label image and folder path
        self.label_image_folder=out_folderpath
        self.label_image_file=out_filepath

    def binarize_image(self):
        '''
        Create binary masks needed to extract dose in each phantom compartment
        '''

        binary_folder_path = os.path.join(args.output, 'Binary_images')
        full_phantom_folder_path = os.path.join(args.output, 'Binary_images', 'PET_Full_phantom')
        # full_phantom_dilate_folder_path = os.path.join(args.output, 'Binary_images', 'PET_Full_phantom_dilate')
        BG_sphere_folder_path = os.path.join(args.output, 'Binary_images', 'PET_BG_sphere')
        cylinder_folder_path = os.path.join(args.output, 'Binary_images', 'PET_Cylinder')
        # cylinder_erode_foler_path = os.path.join(args.output, 'Binary_images', 'PET_Cylinder_erode')
        # BG_MultiLabel_binary_folder_path = os.path.join(args.output, 'Binary_images', 'PET_BG_MultiLabel_binary')
        spheres_folder_path= os.path.join(args.output,'Binary_images','PET_Spheres')
        # phantom_BCK_folder_path=os.path.join(args.output,'Binary_images','PET_phantomBCK')

        ref_dose_spheres_folder_path = os.path.join(args.output, 'Binary_images', 'MC_Spheres')
        ref_dose_full_phantom_folder_path = os.path.join(args.output, 'Binary_images', 'MC_Full_phantom')
        ref_dose_cylinder_folder_path = os.path.join(args.output, 'Binary_images', 'MC_Cylinder')
        # ref_phantom_BCK_folder_path=os.path.join(args.output, 'Binary_images', 'MC_phantomBCK')
        ref_BG_sphere_folder_path = os.path.join(args.output, 'Binary_images', 'MC_BG_sphere')

        if os.path.exists(binary_folder_path):
            shutil.rmtree(binary_folder_path)

        if os.path.exists(full_phantom_folder_path):
            shutil.rmtree(full_phantom_folder_path)

        # if os.path.exists(full_phantom_dilate_folder_path):
        #     shutil.rmtree(full_phantom_dilate_folder_path)
        #
            if os.path.exists(BG_sphere_folder_path):
                shutil.rmtree(BG_sphere_folder_path)

            if os.path.exists(ref_BG_sphere_folder_path):
                shutil.rmtree(ref_BG_sphere_folder_path)

        if os.path.exists(cylinder_folder_path):
            shutil.rmtree(cylinder_folder_path)

        # if os.path.exists(cylinder_erode_foler_path):
        #     shutil.rmtree(cylinder_erode_foler_path)
        #
        # if os.path.exists(BG_MultiLabel_binary_folder_path):
        #     shutil.rmtree(BG_MultiLabel_binary_folder_path)


        if os.path.exists(spheres_folder_path):
            shutil.rmtree(spheres_folder_path)

        if os.path.exists(ref_dose_spheres_folder_path):
            shutil.rmtree(ref_dose_spheres_folder_path)

        if os.path.exists(ref_dose_full_phantom_folder_path):
            shutil.rmtree(ref_dose_full_phantom_folder_path)

        if os.path.exists(ref_dose_cylinder_folder_path):
            shutil.rmtree(ref_dose_cylinder_folder_path)
        #
        # if os.path.exists(phantom_BCK_folder_path):
        #     shutil.rmtree(phantom_BCK_folder_path)
        #
        # if os.path.exists(ref_phantom_BCK_folder_path):
        #     shutil.rmtree(ref_phantom_BCK_folder_path)

        os.makedirs(binary_folder_path)
        os.makedirs(full_phantom_folder_path)
        # os.makedirs(phantom_BCK_folder_path)
        # os.makedirs(full_phantom_dilate_folder_path)
        os.makedirs(BG_sphere_folder_path)
        os.makedirs(cylinder_folder_path)
        # os.makedirs(cylinder_erode_foler_path)
        # os.makedirs(BG_MultiLabel_binary_folder_path)
        os.makedirs(spheres_folder_path)
        os.makedirs(ref_dose_spheres_folder_path)
        os.makedirs(ref_dose_full_phantom_folder_path)
        os.makedirs(ref_dose_cylinder_folder_path)
        # os.makedirs(ref_phantom_BCK_folder_path)
        os.makedirs(ref_BG_sphere_folder_path)

        # Segment the full phantom
        print('\t\tExtracting Whole Phantom.... ')

        #PET Image
        list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                        '-i', self.label_image_file,
                                        '-l', '1',
                                        '-u', '9',
                                        '-o', os.path.join(full_phantom_folder_path,'WholePhantom_PET.mha')]
        sub.run(list_args_clitkBinarizeImage)

        # Reference GATE Dose map
        list_args_clitkBinarizeImage_GATE = ['clitkBinarizeImage',
                                        '-i', args.ref_dose_mask_file,
                                        '-l', '1',
                                        '-u', '9',
                                        '-o', os.path.join(ref_dose_full_phantom_folder_path, 'WholePhantom_MC.mha')]
        sub.run(list_args_clitkBinarizeImage_GATE)

        # Dilate the phantom to verify the normalized background outside
        # print('\t\tDilating phantom to verify the normalized background outside...')
        # for i in range(0, 11):
        #     list_args_clitkMorphoMath = ['clitkMorphoMath',
        #                                  '-i', os.path.join(full_phantom_folder_path,'WholePhantom_PET.mha'),
        #                                  '-t', '1',
        #                                  '-r', str(i),
        #                                  '-o',
        #                                  os.path.join(full_phantom_dilate_folder_path,'Phantom_'+str(i)+'pix_PET.mha')]
        #     sub.run(list_args_clitkMorphoMath)

        # Do the normalized background spheres
        print('\t\tExtracting background sphere...')
        list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                        '-i', self.label_image_file,
                                        '-l', '9',
                                        '-u', '9',
                                        '-o',
                                        os.path.join(BG_sphere_folder_path,'BG_Spheres_PET.mha')]
        sub.run(list_args_clitkBinarizeImage)

        list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                        '-i', args.ref_dose_mask_file,
                                        '-l', '9',
                                        '-u', '9',
                                        '-o',
                                        os.path.join(ref_BG_sphere_folder_path, 'BG_Spheres_MC.mha')]
        sub.run(list_args_clitkBinarizeImage)

        # print('\t\tGenerating MultiLabel background spheres...')
        # list_args_clitkConnectedComponentLabeling = ['clitkConnectedComponentLabeling',
        #     '-i', os.path.join(BG_sphere_folder_path,'BG_PET.mha'),
        #     '-o', os.path.join(BG_sphere_folder_path,'BG_MultiLabel_PET.mha')]
        #
        # sub.run(list_args_clitkConnectedComponentLabeling)

        # Separate 36 spheres
        # print('\t\tSeparating MultiLabel background spheres...')
        # for i in range(1, 37):
        #     list_args_clitkBinarizeImage = ['clitkBinarizeImage',
        #                                     '-i', os.path.join(BG_sphere_folder_path,'BG_MultiLabel_PET.mha'),
        #                                     '-l', str(i),
        #                                     '-u', str(i),
        #                                     '-o',
        #                                     os.path.join(BG_MultiLabel_binary_folder_path,
        #                                                  'BG_sphere' + str(i) + f'_PET.mha')]
        #     sub.run(list_args_clitkBinarizeImage)

        # Segment the cylider
        print('\t\tExtracting Cylinder region...')

        #PET Image mask
        list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                        '-i', self.label_image_file,
                                        '-l', '2',
                                        '-u', '2',
                                        '-o', os.path.join(cylinder_folder_path, f'Cylinder_PET.mha')]
        sub.run(list_args_clitkBinarizeImage)

        #MC Ref dose mask
        list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                        '-i', args.ref_dose_mask_file,
                                        '-l', '2',
                                        '-u', '2',
                                        '-o', os.path.join(ref_dose_cylinder_folder_path, f'Cylinder_MC.mha')]
        sub.run(list_args_clitkBinarizeImage)

        #Erode cylinder ROI
        # print('\t\tEroding Cylinder region...')
        # list_args_clitkMorphoMath = ['clitkMorphoMath',
        #                              '-i', os.path.join(cylinder_folder_path, f'Cylinder_PET.mha'),
        #                              '-t', '0',
        #                              '-r', '2',
        #                              '-o',
        #                              os.path.join(cylinder_erode_foler_path,'Cylinder_erode_'+str(2)+f'pixel_PET.mha')]
        # sub.run(list_args_clitkMorphoMath)

        # Segment 6 spheres
        print('\t\tExtracting sphere regions...')
        for i in range(1, 7):
            # PET Image
            list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                            '-i', self.label_image_file,
                                            '-l', str(i + 2),
                                            '-u', str(i + 2),
                                            '-o', os.path.join(spheres_folder_path, 'Sphere'+str(i)+f'_PET.mha')]
            sub.run(list_args_clitkBinarizeImage)

            # Reference GATE Dose map
            list_args_clitkBinarizeImage_GATE = ['clitkBinarizeImage',
                                            '-i', args.ref_dose_mask_file,
                                            '-l', str(i + 2),
                                            '-u', str(i + 2),
                                            '-o', os.path.join(ref_dose_spheres_folder_path, 'Sphere' + str(i) + f'_MC.mha')]
            sub.run(list_args_clitkBinarizeImage_GATE)

        # Segment phantom background = Full_phantom - spheres - lung cylinder

        # roi_list_PET=[os.path.join(spheres_folder_path,'Sphere1_PET.mha'),
        #           os.path.join(spheres_folder_path,'Sphere2_PET.mha'),
        #           os.path.join(spheres_folder_path,'Sphere3_PET.mha'),
        #           os.path.join(spheres_folder_path,'Sphere4_PET.mha'),
        #           os.path.join(spheres_folder_path,'Sphere5_PET.mha'),
        #           os.path.join(spheres_folder_path,'Sphere6_PET.mha'),
        #           os.path.join(cylinder_folder_path, 'Cylinder_PET.mha')]
        #
        # roi_list_MC = [os.path.join(ref_dose_spheres_folder_path, 'Sphere1_MC.mha'),
        #                 os.path.join(ref_dose_spheres_folder_path, 'Sphere2_MC.mha'),
        #                 os.path.join(ref_dose_spheres_folder_path, 'Sphere3_MC.mha'),
        #                 os.path.join(ref_dose_spheres_folder_path, 'Sphere4_MC.mha'),
        #                 os.path.join(ref_dose_spheres_folder_path, 'Sphere5_MC.mha'),
        #                 os.path.join(ref_dose_spheres_folder_path, 'Sphere6_MC.mha'),
        #                 os.path.join(ref_dose_cylinder_folder_path, 'Cylinder_MC.mha')]
        #
        # print('\t\tExtracting Phantom background ...')
        #
        # for k,(roi_pet,roi_MC) in enumerate(zip(roi_list_PET,roi_list_MC)):
        #
        #     if k==0:
        #         list_args_clitkImageArithm_PET=['clitkImageArithm',
        #                                     '-i',os.path.join(full_phantom_folder_path,'WholePhantom_PET.mha'),
        #                                     '-j',roi_pet,
        #                                     '-t','7',
        #                                     '-o',os.path.join(phantom_BCK_folder_path,'PhantomBCK_PET.mha')]
        #
        #         list_args_clitkImageArithm_MC = ['clitkImageArithm',
        #                                           '-i', os.path.join(ref_dose_full_phantom_folder_path, 'WholePhantom_MC.mha'),
        #                                           '-j', roi_MC,
        #                                           '-t', '7',
        #                                           '-o', os.path.join(ref_phantom_BCK_folder_path, 'MC_PhantomBCK.mha')]
        #
        #         sub.run(list_args_clitkImageArithm_PET)
        #         sub.run(list_args_clitkImageArithm_MC)
        #     else:
        #         list_args_clitkImageArithm_PET=['clitkImageArithm',
        #                                     '-i',os.path.join(phantom_BCK_folder_path,'PhantomBCK_PET.mha'),
        #                                     '-j',roi_pet,
        #                                     '-t', '7',
        #                                     '-o',os.path.join(phantom_BCK_folder_path,'PhantomBCK_PET.mha')]
        #
        #         list_args_clitkImageArithm_MC = ['clitkImageArithm',
        #                                           '-i', os.path.join(ref_phantom_BCK_folder_path, 'MC_PhantomBCK.mha'),
        #                                           '-j', roi_MC,
        #                                           '-t', '7',
        #                                           '-o', os.path.join(ref_phantom_BCK_folder_path, 'MC_PhantomBCK.mha')]
        #
        #         sub.run(list_args_clitkImageArithm_PET)
        #         sub.run(list_args_clitkImageArithm_MC)
        #
        #     time.sleep(1)

        shutil.rmtree(ref_dose_cylinder_folder_path)
        shutil.rmtree(ref_dose_full_phantom_folder_path)
        shutil.rmtree(cylinder_folder_path)
        shutil.rmtree(full_phantom_folder_path)

    def rescale_PET_image(self):
        '''
        Method for rescaling PET image to match the real phantom activity
        :return:
        '''

    def generate_CumAct(self):
        '''
        Convert concentration (Bq/mL) image to cumulative activity image (Bq.s)
        :return:
        '''
        # cumul_act_map_folderpath = os.path.join(args.output, 'Cumul_Act')
        #
        # if not os.path.exists(cumul_act_map_folderpath):
        #     os.makedirs(cumul_act_map_folderpath)

        pet_img = sitk.ReadImage(self.pet_image_file)

        # Extract image information
        pet_img_origin = pet_img.GetOrigin()
        pet_img_spacing = pet_img.GetSpacing()
        pet_img_direction = pet_img.GetDirection()

        # Convert image to array
        self.pet_image_arr = sitk.GetArrayFromImage(pet_img)

        voxel_vol = pet_img.GetSpacing()[0] * pet_img.GetSpacing()[1] * pet_img.GetSpacing()[2]  # PET image voxel volume

        # Convert concentration image (Bq/mL) to activity image(Bq) => A=C * (V/1000)
        pet_img_act_arr = self.pet_image_arr * (voxel_vol / 1000)

        # Convert activity image to cumulated activity image (Acum=A/lambda)
        lambda_phys = math.log(2) / self.radionuc_half_life_sec[self.radionuc]

        self.pet_cum_act_arr = pet_img_act_arr / lambda_phys

        #Extract cumulative activity map for analysis purposes
        # cumul_act_map_filepath=os.path.join(cumul_act_map_folderpath,'Cumul_Act_Map_'+os.path.basename(self.pet_image_file))
        # cumul_act_map=sitk.GetImageFromArray(self.pet_cum_act_arr)
        #
        # cumul_act_map.SetOrigin(pet_img_origin)
        # cumul_act_map.SetSpacing(pet_img_spacing)
        # cumul_act_map.SetDirection(pet_img_direction)
        #
        # sitk.WriteImage(cumul_act_map,cumul_act_map_filepath)

    def calculate_LDM_dosemap(self):
        '''
        Calculate 3D Dose map from PET Image with the LDM method
        :return:
        '''

        LDM_dosemap_folder_path = os.path.join(args.output, 'LDM_DoseMap')

        if not os.path.exists(LDM_dosemap_folder_path):
            os.makedirs(LDM_dosemap_folder_path)

        #Read pet image
        pet_img=sitk.ReadImage(self.pet_image_file)

        #Extract image information
        pet_img_origin=pet_img.GetOrigin()
        pet_img_spacing=pet_img.GetSpacing()
        pet_img_direction=pet_img.GetDirection()

        #Calculate the S-factor (Dose per radioactive decay)
        voxel_vol=pet_img.GetSpacing()[0]*pet_img.GetSpacing()[1]*pet_img.GetSpacing()[2] #PET image voxel volume

        voxel_den=1 # Density of water: 1 g/cm3

        voxel_mass_kg=(voxel_den * voxel_vol)/(1e6) #Calculate voxel mass en kg

        Eavg_J= self.radionuc_mean_energy_MeV[self.radionuc] * 1.602176462e-13 #Average energy released in Joules

        S_factor= Eavg_J/voxel_mass_kg

        print('\t\tS-factor used [Gy/Bq.s]: ', S_factor)
        print(f'\t\tS-factor used [Gy. mL/Bq]: {S_factor*(voxel_vol/1000)*(self.radionuc_half_life_sec[self.radionuc]/math.log(2))}')

        #Generate dosemap from cumulated activity image

        self.dose_map_LDM_arr=self.pet_cum_act_arr*S_factor

        #Convert dosemap to image
        dose_map_LDM=sitk.GetImageFromArray(self.dose_map_LDM_arr) #Convert dosemap to image

        # Set the origin, spacing, and direction of the image to match those of original image
        dose_map_LDM.SetOrigin(pet_img_origin)
        dose_map_LDM.SetSpacing(pet_img_spacing)
        dose_map_LDM.SetDirection(pet_img_direction)

        # Export Dosemap to output folder
        self.LDM_dosemap_file=os.path.join(LDM_dosemap_folder_path,'LDM_DoseMap_'+os.path.split(self.pet_image_file)[-1])
        sitk.WriteImage(dose_map_LDM,self.LDM_dosemap_file)

        print('\t\tLDM Dosemap saved: ',self.LDM_dosemap_file)

    def calculate_DVK_dosemap(self):
        '''
        Calculate a dosemap via the DVK convolution method
        :return:
        '''
        DVK_dosemap_folder_path = os.path.join(args.output, 'DVK_DoseMap')

        if not os.path.exists(DVK_dosemap_folder_path):
            os.makedirs(DVK_dosemap_folder_path)

        #Read pet image
        pet_img=sitk.ReadImage(self.pet_image_file)

        #Extract image information
        pet_img_origin=pet_img.GetOrigin()
        pet_img_spacing=pet_img.GetSpacing()
        pet_img_direction=pet_img.GetDirection()

        #Read Kernel file
        dvk_img=sitk.ReadImage(args.input_path_DVK_file)

        #Extract number of events used for the simulation
        with open(args.input_path_DVK_stat_file, 'r') as f_stat:
            for line in f_stat:
                if "NumberOfEvents" in line:
                    NumOfEvents = line.split("=")[1].split(' ')[1]
                    break

        #Normalise the dvk image by the number of events to have a kernel for a single disintégration
        dvk_img=dvk_img/float(NumOfEvents)

        normalised_DVK_filepath=os.path.join(DVK_dosemap_folder_path,'Normalised_DVK_'+os.path.basename(args.input_path_DVK_file))
        sitk.WriteImage(dvk_img, normalised_DVK_filepath)

        #Convert Kernel from image to array
        dvk_arr=sitk.GetArrayFromImage(dvk_img)

        #Convolve Dose Kernel with Cumulative activity image

        self.dose_map_DVK_arr = signal.convolve(self.pet_cum_act_arr,dvk_arr, mode='same')

        # Convert dosemap to image
        dose_map_DVK = sitk.GetImageFromArray(self.dose_map_DVK_arr)  # Convert dosemap to image

        # Set the origin, spacing, and direction of the image to match those of original image
        dose_map_DVK.SetOrigin(pet_img_origin)
        dose_map_DVK.SetSpacing(pet_img_spacing)
        dose_map_DVK.SetDirection(pet_img_direction)

        # Export Dosemap to output folder
        self.DVK_dosemap_file=os.path.join(DVK_dosemap_folder_path,'DVK_DoseMap_'+os.path.split(self.pet_image_file)[-1])
        sitk.WriteImage(dose_map_DVK,self.DVK_dosemap_file)

        print('\t\tDVK Dosemap saved: ',self.DVK_dosemap_file)

    def generate_ref_dosemap(self):
        '''
        Generate the reference dosemap and scale it to the correct activity
        :return:
        '''

        ref_dosemap_folder_path = os.path.join(args.output, 'MC_DoseMaps')

        if not os.path.exists(ref_dosemap_folder_path):
            os.makedirs(ref_dosemap_folder_path)

        #Automatically determine theoretical activity through a decay correction
        if args.dicom_flag and args.auto_decay: #If autodecay option and Dicom flag are activated

            print('\t\t Theoretical activity calculated automatically by radioactive decay')

            for dcm_folder in os.listdir(args.PET_Image_folder):

                if os.path.basename(dcm_folder)== os.path.basename(self.pet_image_file.rstrip('.mha')): #Check if the dicom folder name matches the current image

                   dcm_files=os.listdir(os.path.join(args.PET_Image_folder,dcm_folder))

                   #Read the first and last images in the dicom pile (This is because for multistep acquisition, the first image doesn't always correspond to the first step)
                   first_dcm_file=pydicom.dcmread(os.path.join(args.PET_Image_folder,dcm_folder,dcm_files[0]))
                   last_dcm_file=pydicom.dcmread(os.path.join(args.PET_Image_folder,dcm_folder,dcm_files[-1]))

                   #Extract calibration information (Activity, Date and Time) => 0018,1074 & 0018,1078 tags
                   A0=first_dcm_file[0x0054,0x0016][0][0x0018,0x1074].value #Initial phantom activity at calibration
                   D0=first_dcm_file[0x0054,0x0016][0][0x0018,0x1078].value[:8] #Calibration date
                   T0=first_dcm_file[0x0054,0x0016][0][0x0018,0x1078].value[8:14] #Calibration time

                   #Extract acquisition date and time from the first and last slice of the dicom pile, only keep the smallest acquisition start time => 0008,002 & 0008,0032 tags
                   D_first=first_dcm_file.AcquisitionDate # Acquisition date on the first image
                   T_first=first_dcm_file.AcquisitionTime # Acquisition time on the first image

                   D_last = last_dcm_file.AcquisitionDate  # Acquisition date on the last image
                   T_last = last_dcm_file.AcquisitionTime  # Acquisition time on the last image

                   #Create datetime objects | Format YYYY,MM,DD,HH,MM,SS
                   calib_datetime=datetime(int(D0[0:4]),int(D0[4:6]),int(D0[6:8]),int(T0[0:2]),int(T0[2:4]),int(T0[4:6]))
                   first_datetime=datetime(int(D_first[0:4]),int(D_first[4:6]),int(D_first[6:8]),int(T_first[0:2]),int(T_first[2:4]),int(T_first[4:6]))
                   last_datetime = datetime(int(D_last[0:4]), int(D_last[4:6]), int(D_last[6:8]), int(T_last[0:2]),int(T_last[2:4]), int(T_last[4:6]))

                   #Compare the two times and choose the earliest one. If they are equal (ex: 1 bed acquisition) chose either one for the future
                   if first_datetime < last_datetime:
                       acquisition_datetime=first_datetime

                   elif first_datetime > last_datetime:
                       acquisition_datetime=last_datetime

                   elif first_datetime == last_datetime:
                       acquisition_datetime = first_datetime

                   # Calculate the time difference in seconds
                   time_diff_sec=(acquisition_datetime - calib_datetime).total_seconds()

                   #Calculate the theoretical activity in MBq at the acquisition start Date and Time
                   lambda_phys = math.log(2) / self.radionuc_half_life_sec[self.radionuc]
                   self.activity_MBq=(A0 * math.exp(-lambda_phys * time_diff_sec))/1e6


        else: #Read theoretical activities from an input csv file
            print('\t\t Reading theoretical activity from input file')
            #Read the theoretical activity input file
            df_th_act=pd.read_csv(args.theoretical_act_file)
            # Extract the activity corresponding to the acquisition date
            self.activity_MBq = df_th_act[df_th_act.iloc[:, 0] == float(self.acq_tag)].iloc[0, 1]

        #Calculate the corresponding cumulative activity
        lambda_phys = math.log(2) / self.radionuc_half_life_sec[self.radionuc]
        cumul_act=(self.activity_MBq *1e6)/lambda_phys

        #Extract number of events used for the simulation of reference dosemap
        with open(args.ref_dose_stat_file, 'r') as f_stat:
            for line in f_stat:
                if "NumberOfEvents" in line:
                    NumOfEvents_dosi = line.split("=")[1].split(' ')[1]
                    break

        # Calculate the scaling factor that needs to be applied
        scaling_factor=cumul_act/float(NumOfEvents_dosi)

        print(f'\t\t Theoretical activity for current timepoint = {self.activity_MBq} MBq')
        print(f'\t\t Resulting cumulative activity = {cumul_act:e} Bq.s')
        print(f'\t\t Number of events in reference dose map= {float(NumOfEvents_dosi):e} ')
        print('\t\t Dose scaling factor: ', scaling_factor)

        #Read reference dose image
        ref_dose_img=sitk.ReadImage(args.input_path_ref_dose_file)

        # Scale the reference dose image
        ref_dose_img = ref_dose_img * scaling_factor

        # Export reference Dosemap to output folder
        self.GATE_dosemap_file = os.path.join(ref_dosemap_folder_path,'MC_DoseMap_' + self.acq_tag+'.mha')
        sitk.WriteImage(ref_dose_img, self.GATE_dosemap_file)

    def generate_DVH(self):
        '''
        Calculate and save the DVH data for LDM, DVK and MC dose maps for each structure of interest (Spheres and phantom background)
        :return:
        '''

        LDM_DVH_folder_path = os.path.join(args.output, 'LDM_DVH')
        DVK_DVH_folder_path = os.path.join(args.output, 'DVK_DVH')
        MC_DVH_folder_path = os.path.join(args.output, 'MC_DVH')

        for dvh_folder in [LDM_DVH_folder_path,DVK_DVH_folder_path,MC_DVH_folder_path]:
            if not os.path.exists(dvh_folder):
                os.makedirs(dvh_folder)

        for binary_mask_folder in os.listdir(os.path.join(args.output,'Binary_Images')):
            for binary_mask in os.listdir(os.path.join(args.output,'Binary_Images',binary_mask_folder)):
                if 'MC' in binary_mask_folder: #If MC mask, use MC dose map

                    list_args_clitkImageStatistics = [
                        'clitkImageStatistics',
                        '-i', self.GATE_dosemap_file,
                        '-m', os.path.join(args.output,'Binary_Images',binary_mask_folder,binary_mask),
                        '-v']
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)
                    output = p0.communicate()[0]
                    output = output.decode()

                    for row in output.splitlines():
                        if "Max" in row:  # Extract maximum value in ROI
                            max_val = row.split(":")[1]
                            max_val = float(max_val)

                        if "Mean" in row:
                            self.GATE_mean_dose.append(float(row.split(":")[1])) #Extract mean dose for structure

                    # Calculate binsize according to max value and number of bins (200)
                    bin_size = (max_val - 0.0) / 200  # 200 = number of bins
                    hdv_max_val = max_val + 5 * bin_size

                    # Generate DVH

                    list_args_clitkImageStatistics = [
                        'clitkImageStatistics',
                        '-i', self.GATE_dosemap_file,
                        '-m', os.path.join(args.output,'Binary_Images',binary_mask_folder,binary_mask),
                        '--dvhistogram', os.path.join(MC_DVH_folder_path,f"{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"),
                        '--centreBin',
                        '--bins', '200',
                        '--lower', '0',
                        '--upper', str(hdv_max_val)]
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)

                    time.sleep(1)  # Wait 1 second before the next loop so that the system has to time to process the sub.Popen

                    self.GATE_DVH_path.append(os.path.join(MC_DVH_folder_path,f"{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"))
                else:
                    # LDM
                    list_args_clitkImageStatistics= [
                         'clitkImageStatistics',
                         '-i', self.LDM_dosemap_file,
                         '-m', os.path.join(args.output, 'Binary_Images', binary_mask_folder, binary_mask),
                         '-v']
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)
                    output = p0.communicate()[0]
                    output = output.decode()

                    for row in output.splitlines():
                        if "Max" in row:  # Extract maximum value in ROI
                            max_val = row.split(":")[1]
                            max_val = float(max_val)

                    # Calculate binsize according to max value and number of bins (200)
                    bin_size = (max_val - 0.0) / 200  # 200 = number of bins
                    hdv_max_val = max_val + 5 * bin_size

                    list_args_clitkImageStatistics = [
                        'clitkImageStatistics',
                        '-i', self.LDM_dosemap_file,
                        '-m', os.path.join(args.output, 'Binary_Images', binary_mask_folder, binary_mask),
                        '--dvhistogram', os.path.join(LDM_DVH_folder_path,f"LDM_{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"),
                        '--centreBin',
                        '--bins', '200',
                        '--lower', '0',
                        '--upper', str(hdv_max_val)]
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)


                    time.sleep(1)  # Wait 1 second before the next loop so that the system has to time to process the sub.Popen

                    self.LDM_DVH_path.append(os.path.join(LDM_DVH_folder_path,f"LDM_{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"))

                    # DVK

                    list_args_clitkImageStatistics = [
                        'clitkImageStatistics',
                        '-i', self.DVK_dosemap_file,
                        '-m', os.path.join(args.output, 'Binary_Images', binary_mask_folder, binary_mask),
                        '-v']
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)
                    output = p0.communicate()[0]
                    output = output.decode()

                    for row in output.splitlines():
                        if "Max" in row:  # Extract maximum value in ROI
                            max_val = row.split(":")[1]
                            max_val = float(max_val)

                    # Calculate binsize according to max value and number of bins (200)
                    bin_size = (max_val - 0.0) / 200  # 200 = number of bins
                    hdv_max_val = max_val + 5 * bin_size

                    list_args_clitkImageStatistics = [
                        'clitkImageStatistics',
                        '-i', self.DVK_dosemap_file,
                        '-m', os.path.join(args.output, 'Binary_Images', binary_mask_folder, binary_mask),
                        '--dvhistogram', os.path.join(DVK_DVH_folder_path,f"DVK_{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"),
                        '--centreBin',
                        '--bins', '200',
                        '--lower', '0',
                        '--upper', str(hdv_max_val)]
                    p0 = sub.Popen(list_args_clitkImageStatistics, stdout=sub.PIPE)

                    time.sleep(1)  # Wait 1 second before the next loop so that the system has to time to process the sub.Popen
                    self.DVK_DVH_path.append(os.path.join(DVK_DVH_folder_path,f"DVK_{binary_mask.rstrip('.mha')}_{os.path.splitext(os.path.basename(self.pet_image_file))[0]}.csv"))

    def plot_DVH(self):
        '''
        Plot LDM, DVK and MC derived DVH on the same curve for each PET image and each sphere structure
        :return:
        '''

        DVH_plot_path = os.path.join(args.output, 'DVH_plot')
        if not os.path.exists(DVH_plot_path):
            os.makedirs(DVH_plot_path)


        for k, (ldm_dvh_structure,dvk_dvh_structure,gate_dvh_structure) in enumerate(zip(self.LDM_DVH_path,self.DVK_DVH_path,self.GATE_DVH_path)):

            structure_name=os.path.splitext(os.path.basename(ldm_dvh_structure))[0].split('_')[1]

            df_ldm = pd.read_csv(ldm_dvh_structure, header=5, sep='\t',
                             names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                    'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                    'Volume_cumul[cc]'])

            df_dvk=pd.read_csv(dvk_dvh_structure, header=5, sep='\t',
                             names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                    'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                    'Volume_cumul[cc]'])

            df_gate = pd.read_csv(gate_dvh_structure, header=5, sep='\t',
                                 names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                        'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                        'Volume_cumul[cc]'])

            dose_Gy_ldm=df_ldm['Dose_diff[Gy]'].values
            vol_cum_rel_ldm=df_ldm['Volume_cumul[%]'].values

            dose_Gy_dvk = df_dvk['Dose_diff[Gy]'].values
            vol_cum_rel_dvk = df_dvk['Volume_cumul[%]'].values

            dose_Gy_GATE = df_gate['Dose_diff[Gy]'].values
            vol_cum_rel_GATE = df_gate['Volume_cumul[%]'].values

            plt.figure()
            plt.plot(dose_Gy_ldm,vol_cum_rel_ldm,'b-', linewidth=2, markersize=2.0,label='LDM')
            plt.plot(dose_Gy_dvk, vol_cum_rel_dvk, 'r-', linewidth=2, markersize=2.0, label='DVK')
            plt.plot(dose_Gy_GATE,vol_cum_rel_GATE,'g-',linewidth=2, markersize=2.0, label='MC')
            plt.suptitle(f'DVH for {structure_name}')
            plt.title(os.path.split(self.pet_image_file)[-1].split('.mha')[0])
            plt.legend()
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Volume (%)')
            plt.ylim([0,103])
            plt.xlim([0,None])
            plt.grid()
            # plt.show()
            output_file_name=f'DVH_{structure_name}_'+os.path.split(self.pet_image_file)[-1].split('.mha')[0]+'.pdf'
            plt.savefig(os.path.join(DVH_plot_path,output_file_name))
            plt.close()

    def calculate_rmse(self):
        '''
        Generate an inverse DVH and perform RMSE calculation
        :return: Dictionary containing rmse values for each sphere
        '''

        invDVH_plot_path = os.path.join(args.output, 'invDVH_plot')
        if not os.path.exists(invDVH_plot_path):
            os.makedirs(invDVH_plot_path)

        for k, (ldm_dvh_structure,dvk_dvh_structure,gate_dvh_structure,gate_mean_dose_structure) in enumerate(zip(self.LDM_DVH_path,self.DVK_DVH_path,self.GATE_DVH_path,self.GATE_mean_dose)):
            structure_name = os.path.splitext(os.path.basename(ldm_dvh_structure))[0].split('_')[1]
            df_ldm = pd.read_csv(ldm_dvh_structure, header=5, sep='\t',
                             names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                    'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                    'Volume_cumul[cc]'])

            df_dvk=pd.read_csv(dvk_dvh_structure, header=5, sep='\t',
                             names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                    'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                    'Volume_cumul[cc]'])

            df_gate = pd.read_csv(gate_dvh_structure, header=5, sep='\t',
                                 names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                        'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                        'Volume_cumul[cc]'])

            #Interpolate y values for the DVHs
            interp_LDM=interp1d(df_ldm['Dose_diff[Gy]'], df_ldm['Volume_cumul[%]'],bounds_error=False, fill_value=(100, 0))
            interp_DVK=interp1d(df_dvk['Dose_diff[Gy]'], df_dvk['Volume_cumul[%]'],bounds_error=False, fill_value=(100, 0))
            interp_MC=interp1d(df_gate['Dose_diff[Gy]'], df_gate['Volume_cumul[%]'],bounds_error=False, fill_value=(100, 0))

            #Reverse the DVH function
            invLDM_x,invLDM_y=invDistributionFunction(df_ldm['Dose_diff[Gy]'].values,interp_LDM)
            invDVK_x, invDVK_y = invDistributionFunction(df_dvk['Dose_diff[Gy]'].values, interp_DVK)
            invMC_x, invMC_y = invDistributionFunction(df_gate['Dose_diff[Gy]'].values, interp_MC)

            #Interpolate reversed DVH curves
            interp_invLDM=interp1d(invLDM_x,invLDM_y,bounds_error=False, fill_value=(100, 0))
            interp_invDVK=interp1d(invDVK_x,invDVK_y,bounds_error=False, fill_value=(100, 0))
            interp_invMC=interp1d(invMC_x,invMC_y,bounds_error=False, fill_value=(100, 0))

            #Calculate RMSE (Root Mean Square Error)
            xDVH = np.linspace(0, 100, 1001) # 1001 samples between 0 and 100 % (domain of inverse DVH)
            rmse_LDM=(dvhRMSE(xDVH,interp_invLDM,interp_invMC))
            rmse_DVK=(dvhRMSE(xDVH,interp_invDVK,interp_invMC))

            #Fill RMSE dictionary
            self.nrmse_LDM['Name']=os.path.basename(self.pet_image_file).split('.mha')[0]
            self.nrmse_LDM[f'nRMSE {structure_name}']= round(rmse_LDM/gate_mean_dose_structure, 3)

            self.nrmse_DVK['Name'] = os.path.basename(self.pet_image_file).split('.mha')[0]
            self.nrmse_DVK[f'nRMSE {structure_name}'] = round(rmse_DVK/gate_mean_dose_structure, 3)

            self.rmse_LDM['Name'] = os.path.basename(self.pet_image_file).split('.mha')[0]
            self.rmse_LDM[f'RMSE {structure_name}'] = round(rmse_LDM, 3)

            self.rmse_DVK['Name'] = os.path.basename(self.pet_image_file).split('.mha')[0]
            self.rmse_DVK[f'RMSE {structure_name}'] = round(rmse_DVK, 3)

            # Plot inverse DVH
            plt.figure()
            plt.plot(interp_invLDM.x, interp_invLDM.y, 'b-', linewidth=2, markersize=2.0, label='LDM')
            plt.plot(interp_invDVK.x, interp_invDVK.y, 'r-', linewidth=2, markersize=2.0, label='DVK')
            plt.plot(interp_invMC.x, interp_invMC.y, 'g-', linewidth=2, markersize=2.0, label='MC')
            plt.suptitle(f'Inverse DVH for {structure_name}')
            plt.title(os.path.split(self.pet_image_file)[-1].split('.mha')[0])
            plt.legend()
            plt.xlabel('Volume (%)')
            plt.ylabel('Dose (Gy)')
            plt.xlim([0, 103])
            plt.grid()
            # plt.show()
            output_file_name = f'invDVH_{structure_name}_' + os.path.split(self.pet_image_file)[-1].split('.mha')[0] + '.pdf'
            plt.savefig(os.path.join(invDVH_plot_path, output_file_name))
            plt.close()

    def compare_DVH_with_MIM(self):
        '''
        Bonus: Compare Python generated DVHs with MIM for DLM and for DVK (VSV)
        :return:
        '''
        df_mim_LDM=pd.read_csv('./MIM_vs_Python_comp/LDM/ACQ1-NEMAY90_LDM_DVH_MIM.csv',skiprows=1)
        df_mim_DVK=pd.read_csv('./MIM_vs_Python_comp/DVK/ACQ1-NEMAY90_VSV_DVH_MIM.csv',skiprows=1)


        # df_mim=df_mim.dropna()

        dose_Gy_mim_LDM=df_mim_LDM['Dose absolue(Gy)'].values
        vol_s1_mim_LDM=df_mim_LDM['S1 (40) (Volume: 0,52318 mL)'].values
        vol_s2_mim_LDM=df_mim_LDM['S2 (41) (Volume: 1,1632 mL)'].values
        vol_s3_mim_LDM=df_mim_LDM['S3 (42) (Volume: 2,5158 mL)'].values
        vol_s4_mim_LDM=df_mim_LDM['S4 (43) (Volume: 5,5029 mL)'].values
        vol_s5_mim_LDM=df_mim_LDM['S5 (44) (Volume: 11,381 mL)'].values
        vol_s6_mim_LDM=df_mim_LDM['S6 (28) (Volume: 26,558 mL)'].values


        dose_Gy_mim_DVK=df_mim_DVK['Absolute Dose(Gy)'].values
        vol_s1_mim_DVK=df_mim_DVK['S1 (10) (Volume: 0.52318 mL)'].values
        vol_s2_mim_DVK=df_mim_DVK['S2 (11) (Volume: 1.1632 mL)'].values
        vol_s3_mim_DVK=df_mim_DVK['S3 (12) (Volume: 2.5158 mL)'].values
        vol_s4_mim_DVK=df_mim_DVK['S4 (13) (Volume: 5.5029 mL)'].values
        vol_s5_mim_DVK=df_mim_DVK['S5 (14) (Volume: 11.381 mL)'].values
        vol_s6_mim_DVK=df_mim_DVK['S6 (3) (Volume: 26.558 mL)'].values

        vol_sphere_mim_LDM=np.array([vol_s1_mim_LDM,vol_s2_mim_LDM,vol_s3_mim_LDM,vol_s4_mim_LDM,vol_s5_mim_LDM,vol_s6_mim_LDM])

        vol_sphere_mim_DVK = np.array([vol_s1_mim_DVK, vol_s2_mim_DVK, vol_s3_mim_DVK, vol_s4_mim_DVK, vol_s5_mim_DVK, vol_s6_mim_DVK])

        if '31032022' in self.pet_image_file:
            for k, (dvh_sphere_ldm,dvh_sphere_dvk) in enumerate(zip(self.LDM_DVH_path,self.DVK_DVH_path)):
                df_dlm = pd.read_csv(dvh_sphere_ldm, header=5, sep='\t',
                                 names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                        'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                        'Volume_cumul[cc]'])

                df_dvk = pd.read_csv(dvh_sphere_dvk, header=5, sep='\t',
                                     names=['Dose_diff[Gy]', 'Volume_diff[N°voxels]', 'Volume_diff[%]',
                                            'Volume_diff[cc]', 'Volume_cumul[N°voxels]', 'Volume_cumul[%]',
                                            'Volume_cumul[cc]'])

                dose_Gy_ldm=df_dlm['Dose_diff[Gy]'].values
                vol_cum_rel_ldm=df_dlm['Volume_cumul[%]'].values

                dose_Gy_dvk=df_dvk['Dose_diff[Gy]'].values
                vol_cum_rel_dvk=df_dvk['Volume_cumul[%]'].values

                plt.figure()
                plt.plot(dose_Gy_ldm,vol_cum_rel_ldm,'b-', linewidth=2, markersize=2.0,label='Python')
                plt.plot(dose_Gy_mim_LDM,vol_sphere_mim_LDM[k],'r-', linewidth=2, markersize=2.0,label='MIM')
                plt.title(os.path.split(dvh_sphere_ldm)[-1])
                plt.suptitle('LDM - Same ROIs')
                plt.xlabel('Dose (Gy)')
                plt.ylabel('Volume (%)')
                plt.legend()
                plt.xlim([0,None])
                plt.ylim([0,103])
                plt.grid()
                plt.savefig(f'./MIM_vs_Python_comp/LDM/DVH_with_MIM_ROIs/Sphere{k+1}_SameROIs.pdf')

                plt.figure()
                plt.plot(dose_Gy_dvk, vol_cum_rel_dvk, 'b-', linewidth=2, markersize=2.0, label='Python')
                plt.plot(dose_Gy_mim_DVK, vol_sphere_mim_DVK[k], 'r-', linewidth=2, markersize=2.0, label='MIM')
                plt.title(os.path.split(dvh_sphere_dvk)[-1])
                plt.suptitle('DVK - Same ROIs')
                plt.xlabel('Dose (Gy)')
                plt.ylabel('Volume (%)')
                plt.legend()
                plt.xlim([0, np.max(dose_Gy_ldm) + 2])
                plt.ylim([0, 103])
                plt.grid()
                plt.savefig(f'./MIM_vs_Python_comp/DVK/DVH_with_MIM_ROIs/Sphere{k+1}_SameROIs.pdf')



# ==================================M A I N=====================================================================
def main():

    #Create output folder if it doesn't exist
    if not os.path.exists(args.output):
        print('Creating output folder: ',args.output)
        os.makedirs(args.output)

    # Instantiation of new objects which will be stored in the image_object_list
    image_object_list = instantiate_object()

    # Check for unmatched PET images
    if len(image_object_list) != len(os.listdir(args.PET_Image_folder)):
        # Store the images in the input folder and in the object list into two sets

        image_files_set = set(os.listdir(args.PET_Image_folder))
        image_object_set = set([])

        # Fill the image_object_set with the names of the files in the image object list
        for k in range(len(image_object_list)):
            image_object_set.add(os.path.split(image_object_list[k].pet_image_file)[-1])

        # Identify the files that don't match between the two (images that have no corresponding label image)
        images_with_no_mask = list(image_files_set - image_object_set)

        raise Exception(f"The following PET images have no corresponding label image: {'|'.join(images_with_no_mask)}")
        # print('Warning: One or more PET images have no corresponding label image. They will not be analysed')


    if (args.dicom_flag):

        if os.path.exists(os.path.join(args.output, 'mha_images')):
            shutil.rmtree(os.path.join(args.output, 'mha_images'))

        os.makedirs(os.path.join(args.output, 'mha_images'))

    if (args.affine_transform_flag):
        if os.path.exists(os.path.join(os.path.split(os.path.abspath(args.label_image_folder))[-2], 'PET_sized_label_images')):
            shutil.rmtree(os.path.join(os.path.split(os.path.abspath(args.label_image_folder))[-2], 'PET_sized_label_images'))

        os.makedirs(os.path.join(os.path.split(os.path.abspath(args.label_image_folder))[-2], 'PET_sized_label_images'))


    #Check the system Os to define the destination for temporary file saving
    if platform.system()=='Windows':
        dst='C:/Coupes'
    else:
        dst='/tmp/Coupes'

    # Initialise dataframe that will contain all rmse values for different reconstructions
    df_nrmse_LDM=pd.DataFrame()
    df_nrmse_DVK=pd.DataFrame()

    df_rmse_LDM = pd.DataFrame()
    df_rmse_DVK = pd.DataFrame()

    # Loop through every image object
    for image_object in image_object_list:

        # Step 1: Check if DICOM flag is active

        if (image_object.dicom_flag):  # Convert DICOM images to mha
            print('Converting PET Image from DICOM to mha')
            image_object.dicom2Image(dst)

        # Step 2: Check if the affine transform flag is active
        if (image_object.affine_transform_flag):  # Resample CT sized labelled image to PET size
            print('Generating PET-sized Labelled Image')
            image_object.affine_transform()

        print('\nAnalaysing: ', os.path.split(image_object.pet_image_file)[-1])

        print('\tChosen Mask image: ',image_object.label_image_file)
        #
        # Step 3: Create binary masks for the hot and BG spheres
        print('\n\tCreating Binary Masks')
        image_object.binarize_image()

        #Step 4: Convert Concentration image to cumulative activity
        image_object.generate_CumAct()

        # Step 5: Calculate dose by LDM
        print('\n\tCalculating LDM Dose map')
        image_object.calculate_LDM_dosemap()

        # Step 6: Calculate dose by DVK
        print('\n\tCalculating DVK Dose map')
        image_object.calculate_DVK_dosemap()

        # Step 7: Extract reference dose map
        print('\n\tGenerating reference dose map')
        image_object.generate_ref_dosemap()
        #
        # Step 8: Extract DVH
        print('\n\tExtracting DVH curve data')
        image_object.generate_DVH()
        #
        # Step 9: Plot cumulative DVH curve
        print('\n\tPlotting DVH curves')
        image_object.plot_DVH()

        # Step 10: Calculate inverse DVH
        print('\n\tCalculating inverse DVH')
        image_object.calculate_rmse()

        # Step 11: Update RMSE dataframes
        print('\n\tUpdating RMSE table... ')
        df_nrmse_LDM=pd.concat([df_nrmse_LDM,pd.DataFrame([image_object.nrmse_LDM])])
        df_nrmse_DVK=pd.concat([df_nrmse_DVK,pd.DataFrame([image_object.nrmse_DVK])])

        df_rmse_LDM = pd.concat([df_rmse_LDM, pd.DataFrame([image_object.rmse_LDM])])
        df_rmse_DVK = pd.concat([df_rmse_DVK, pd.DataFrame([image_object.rmse_DVK])])

        # Delete Binary_images folder from output folder
        shutil.rmtree(os.path.join(args.output, 'Binary_images'))


        # image_object.compare_DVH_with_MIM()


    #Export RMSE Dataframes
    rmse_export_path = os.path.join(args.output, 'RMSE')
    if not os.path.exists(rmse_export_path):
        os.makedirs(rmse_export_path)

    print('\n\tExtracting RMSE table')
    df_nrmse_LDM.to_csv(os.path.join(rmse_export_path, f'nRMSE_LDM_{os.path.basename(os.path.abspath(args.PET_Image_folder))}.csv'), index=False)
    df_nrmse_DVK.to_csv(os.path.join(rmse_export_path, f'nRMSE_DVK_{os.path.basename(os.path.abspath(args.PET_Image_folder))}.csv'), index=False)

    df_rmse_LDM.to_csv(os.path.join(rmse_export_path, f'RMSE_LDM_{os.path.basename(os.path.abspath(args.PET_Image_folder))}.csv'),index=False)
    df_rmse_DVK.to_csv(os.path.join(rmse_export_path, f'RMSE_DVK_{os.path.basename(os.path.abspath(args.PET_Image_folder))}.csv'),index=False)

    #Plot RMSE graphs

    print('\n\tPlotting RMSE graphs')
    plot_rmse(df_nrmse_LDM,df_nrmse_DVK)
    plot_rmse(df_rmse_LDM, df_rmse_DVK)

if __name__ == "__main__":
    main()