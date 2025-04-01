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
import csv
from scipy.stats import chisquare
# =============================================P A R S E R=====================================================
parser = argparse.ArgumentParser(description='Estimate the tomographic spatial resolution from an IEC acquisiton')

parser.add_argument('--label_image_folder', type=str, help='Path towards folder containing the labelled image containing the ROIs')
parser.add_argument('--input_image_folder', type=str,
                    help='Path towards the folder containing the input images (Dicom folders or standard image file)')

parser.add_argument('--input_blurred_folder',type=str,help='Path towards the folder containing the  blurred images')
parser.add_argument('--label_blurred_file',type=str,help='Path towards the labelled image for the blurred images')

parser.add_argument('-dc', '--dicom_flag', help='Include this option if the static PET is in DICOM format',
                    action='store_true')
parser.add_argument('--radionuc',type=str,help='Radionucleide (options: Y90)')
parser.add_argument('--sphere_con',type=float,help='Theoretical sphere concentration in kBq/mL at calibration time or acquisition time')

parser.add_argument('-con_rat', '--concentration_ratio', type=float,help='Enter the theoretical Sphere-to-background ratio (the one present in the ideal image)')
parser.add_argument('-auto_decay', help='Include this option if sphere concentration was given at calibration time and you want to auto decay correct to acqusition time', action='store_true')
parser.add_argument('-auto_scale','--auto_scale', help='Include this option to automatically rescale image to match theoretical activity', action='store_true')
parser.add_argument('-Th_Act_file','--theoretical_act_file',type=str,help='path to the csv file containing the theoretical activities in MBq (Include if auto_decay is off)')
parser.add_argument('-o', '--output', type=str, help='Path to output folder')

args = parser.parse_args()


def instantiate_PETobject():
    '''
    Instantiate new image object into a list
    :return: image_object_list
    '''
    image_object_list = []

    for image_file in os.listdir(args.input_image_folder):
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

def instantiate_Blurredobject():
    '''
    Instantiate new blurred image object into a list
    :return: blurredimage_object_list
    '''

    blurredimage_object_list=[]

    for blurredimage in os.listdir(args.input_blurred_folder):
        blurredimage_object_list.append(BlurredImage(blurredimage))


    return blurredimage_object_list

def rmse_comp(dataA,dataB):
    MSE = np.square(np.subtract(dataA, dataB)).mean()
    RMSE = math.sqrt(MSE)

    return RMSE

def plot_rmse_vs_fwhm(df):

    output_folder=os.path.join(args.output,'Plots')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pet_image_group=df.groupby('Image Name')
    for image_name,group in pet_image_group:
        #RMSE plot
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(group['Blurred image FWHM'], group['RMSE'],'.',markersize=1.5)
        plt.xlabel('FWHM (mm)')
        plt.ylabel('RMSE')
        plt.title(f'RMSE vs FWHM for {image_name}')
        plt.savefig(os.path.join(output_folder, f"RMSEvsFWHM_{image_name}.pdf"), bbox_inches='tight')
        plt.close()


def estimate_RS(image_object_list,blurred_image_object_list):

    df_rs_analysis=pd.DataFrame(columns=['Image Name','Blurred image FWHM','RMSE'])

    for pet_image in image_object_list:
        fwhm=[]
        rmse=[]
        for blurred_image in blurred_image_object_list:

            # RMSE Comparaison
            RMSE = rmse_comp(blurred_image.RC,pet_image.RC)

            fwhm.append(blurred_image.fwhm_mm)
            rmse.append(RMSE)

            new_row = pd.DataFrame([[pet_image.imagename,blurred_image.fwhm_mm,RMSE]],columns=df_rs_analysis.columns)
            df_rs_analysis = pd.concat([df_rs_analysis, new_row], ignore_index=True)

        pet_image.SR=fwhm[rmse.index(min(rmse))]


    df_rs_analysis_sorted = df_rs_analysis.sort_values(by=['Image Name', 'Blurred image FWHM'], ascending=[True, True])

    # Find the FWHM values that minimise RMSE
    idx = df_rs_analysis_sorted.groupby('Image Name')['RMSE'].idxmin()
    min_rmse_rows = df_rs_analysis_sorted.loc[idx]

    df_rs_analysis_sorted.to_csv(os.path.join(args.output,f'RC_comparison.csv'),index=False)

    #Print and export
    print('*' * 100)
    print('Estimated spatial resolution')
    for index, row in min_rmse_rows.iterrows():
        # Extract the values for 'Name', 'FWHM', and 'RMSE' from the current row
        name = row['Image Name']
        fwhm = row['Blurred image FWHM']
        rmse = row['RMSE']

        # Print the desired text with values inserted
        print(f"Estimation spatial resolution for {name}: {fwhm} mm (RMSE = {round(rmse, 3)})")


    # Export to txt file
    with open(os.path.join(args.output, 'RS_results.txt'), 'w') as file:

        print('*' * 100,file=file)

        for index, row in min_rmse_rows.iterrows():
            # Extract the values for 'Name', 'FWHM', and 'RMSE' from the current row
            name = row['Image Name']
            fwhm = row['Blurred image FWHM']
            rmse = row['RMSE']

            # Print the desired text with values inserted
            print(f"Estimation spatial resolution for {name}: {fwhm} mm (RMSE = {round(rmse,3)})",file=file)
    return df_rs_analysis_sorted


def plot_rc_comp(image_object_list):
    '''
    Plot CRC vs Sphere diameter curve for all the input images on the same graph
    :param image_object_list:
    '''
    output_folder = os.path.join(args.output, 'Plots')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sphere_diam = np.array([10, 13, 17, 22, 28, 37])
    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.grid()
    for image_object in image_object_list:
        plt.plot(sphere_diam,image_object.RC,'.--', label=image_object.imagename)
        plt.ylim([0, 1])
        plt.xticks([10, 13, 17, 22, 28, 37])
        plt.title(f"RC vs Sphere Diameter")
        plt.xlabel('Sphere Diameter (mm)')
        plt.ylabel('Recovery Coefficient')
        plt.tight_layout()

    fig.legend(bbox_to_anchor=(1.5, 0.5), borderaxespad=0)

    plt.savefig(os.path.join(output_folder, f"RCvsSphereDiam_Comp.pdf"),bbox_inches='tight')
    plt.close()


def plot_pet_blurred_rc_comp(image_object_list,blurred_image_object_list):
    sphere_diam = np.array([10, 13, 17, 22, 28, 37])
    output_folder = os.path.join(args.output, 'Plots')
    for pet_image in image_object_list:
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.grid()
        for blurred_image in blurred_image_object_list:
            if pet_image.SR == blurred_image.fwhm_mm: #If image spatial resolution matches blurred image fwhm
                plt.plot(sphere_diam, pet_image.RC, 'k.--', label=pet_image.imagename)
                plt.plot(sphere_diam,blurred_image.RC,'b.--',label=f'Blurred FWHM {blurred_image.fwhm_mm}')
                plt.ylim([0, 1])
                plt.xticks([10, 13, 17, 22, 28, 37])
                plt.title(f"PET image vs Blurred image RC comparison")
                plt.xlabel('Sphere Diameter (mm)')
                plt.ylabel('Recovery Coefficient')
                plt.tight_layout()

        fig.legend(bbox_to_anchor=(1.3, 0.91), borderaxespad=0)

        plt.savefig(os.path.join(output_folder, f"PETvsBlurred_RC_Comp_{pet_image.imagename}.pdf"), bbox_inches='tight')
        plt.close()




class PetImage():
    # Class object attributes (attributes in common for all objects of the class)
    dicom_flag= args.dicom_flag
    label_image_folder = args.label_image_folder
    image_folder = args.input_image_folder
    radionuc_half_life_sec = {'Y90': 230549.76,'F18': 6583.68,'Ga68': 4069.8}
    radionuc_mean_energy_MeV={'Y90':0.935} #MeV

    df_stat_image=pd.DataFrame(columns=['Name','VOI','Volume (cc)','Min','Max','Mean','Real con (kBq/mL)','RC']) #Dataframe that will store image statistics

    def __init__(self,image_file,label_image_file):
        self.image_file=os.path.join(self.image_folder,image_file) # PET Image File path
        self.label_image_file=os.path.join(self.label_image_folder,label_image_file) # Label file path
        self.imagename=os.path.splitext(os.path.basename(self.image_file))[0]
        self.radionuc = args.radionuc  # Radionucleide
        self.acq_tag=os.path.split(self.image_file)[-1].split('_')[-1].rstrip('.mha') #Acquisition tag name extracted from image name
        self.sphere_con_kBqmL=args.sphere_con
        #Initialise attributes
        self.A_image_MBq = 0  # Activity measured in the image
        self.A_th_MBq = None# Theoretical activity
        self.C_moy_kBqmL=np.zeros(6) # Average concentration in the spheres extracted from the image
        self.sphere_vol_cc=np.zeros(6) # Sphere volume extracted through binary ROI
        self.RC=np.zeros(6) #Recovery coefficients will be stored here
        self.data=pd.DataFrame({
            'Sphere Diameter (mm)': [10, 13, 17, 22, 28, 37],
            'Sphere Volume (cc)': [0, 0, 0, 0, 0, 0],
            'RC': [0, 0, 0, 0, 0, 0]
        })
        self.SR = None # Spatial resolution


    def dicom2Image(self,dst):
        '''
         Convert a series of Dicom images into mha format
        :return:
        '''

        #dst = 'C:/Coupes'  # A temporary directory where we will move the DICOM series in order to have a shorter filepath

        if os.path.exists(dst):
            shutil.rmtree(os.path.join(dst))

        shutil.move(self.image_file, dst)

        dirlist = []  # We create an empty list to store the path to all the DICOM files in a series
        [dirlist.append(os.path.abspath(os.path.join(dst, name))) for name in
         os.listdir(dst)]  # We store the path to all the Dicom files (files ending with dcm)
        dirlist = natsorted(dirlist)  # We sort the list by alphabetical/numerical order

        command = ' '.join(dirlist)  # Extract the contents of the dirlist as one string to be used in the clitkDicom2Image function

        output_file_name = os.path.split(self.image_file)[-1] + '.mha'
        output_file_path=os.path.join(args.output, 'mha_images',output_file_name)

        sub.run(f'clitkDicom2Image {command} -o {output_file_path} -t 0.001 -p',shell=True)  # Using a clitk function to perform the Dicom to Image transformation

        shutil.move(dst, self.image_file)

        self.image_folder=os.path.join(os.path.split(os.path.abspath(args.input_image_folder))[-2], 'mha_images')

        self.image_file=output_file_path

    def binarize_image(self):
        '''
        Create binary masks needed to extract dose in each phantom compartment
        '''

        binary_folder_path = os.path.join(args.output, 'Binary_images')

        # BG_sphere_folder_path = os.path.join(args.output, 'Binary_images', 'BG_sphere')

        spheres_folder_path = os.path.join(args.output, 'Binary_images', 'Hot_Spheres')



        if os.path.exists(binary_folder_path):
            shutil.rmtree(binary_folder_path)
        #
        # if os.path.exists(BG_sphere_folder_path):
        #     shutil.rmtree(BG_sphere_folder_path)


        if os.path.exists(spheres_folder_path):
            shutil.rmtree(spheres_folder_path)


        os.makedirs(binary_folder_path)

        # os.makedirs(BG_sphere_folder_path)

        os.makedirs(spheres_folder_path)

        # # Do the normalized background spheres
        # print('\t\tExtracting background sphere...')
        # list_args_clitkBinarizeImage = ['clitkBinarizeImage',
        #                                 '-i', self.label_image_file,
        #                                 '-l', '9',
        #                                 '-u', '9',
        #                                 '-o',
        #                                 os.path.join(BG_sphere_folder_path, 'BG_Spheres.mha')]
        # sub.run(list_args_clitkBinarizeImage)
        #


        # Segment 6 spheres
        print('\t\tExtracting sphere regions...')
        for i in range(1, 7):
            list_args_clitkBinarizeImage = ['clitkBinarizeImage',
                                            '-i', self.label_image_file,
                                            '-l', str(i + 2),
                                            '-u', str(i + 2),
                                            '-o', os.path.join(spheres_folder_path, 'Sphere' + str(i) + '.mha')]
            sub.run(list_args_clitkBinarizeImage)

    def extract_theoretical_act(self):
        '''
        Determine the theoretical activity at time of acquisition through inpu csv file or through the DICOM tags
        :return:
        '''


        if args.dicom_flag and args.auto_scale:

            print('\t\t Theoretical activity calculated automatically by radioactive decay')

            for dcm_folder in os.listdir(args.input_image_folder):

                if os.path.basename(dcm_folder)== os.path.basename(self.image_file.rstrip('.mha')): #Check if the dicom folder name matches the current image

                   dcm_files=os.listdir(os.path.join(args.input_image_folder,dcm_folder))

                   #Read the first and last images in the dicom pile (This is because for multistep acquisition, the first image doesn't always correspond to the first step)
                   first_dcm_file=pydicom.dcmread(os.path.join(args.input_image_folder,dcm_folder,dcm_files[0]))
                   last_dcm_file=pydicom.dcmread(os.path.join(args.input_image_folder,dcm_folder,dcm_files[-1]))

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
                   self.A_th_MBq=(A0 * math.exp(-lambda_phys * time_diff_sec))/1e6

        else: #Read theoretical activities from an input csv file
            print('\t\t Reading theoretical activity from input file')
            #Read the theoretical activity input file
            df_th_act=pd.read_csv(args.theoretical_act_file)
            # Extract the activity corresponding to the acquisition date
            self.A_th_MBq = df_th_act[df_th_act.iloc[:, 0] == float(self.acq_tag)].iloc[0, 1]

    def rescale_PET_image(self):
        '''
        Method for rescaling image to match the real phantom activity. We do this
        :return:
        '''

        self.image_folder = os.path.join(args.output, 'rescaled_PET')

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        #Load PET
        pet_img=sitk.ReadImage(self.image_file)
        # Extract image information
        pet_img_origin = pet_img.GetOrigin()
        pet_img_spacing = pet_img.GetSpacing()
        pet_img_direction = pet_img.GetDirection()

        # Extract pixel volume
        pxl_vol = pet_img.GetSpacing()[0] * pet_img.GetSpacing()[1] * pet_img.GetSpacing()[2]

        #Convert to array
        pet_img_arr=sitk.GetArrayFromImage(pet_img)

        # Convert concentration values to activity A= C*(V/1000)
        pet_img_act_arr = pet_img_arr * (pxl_vol / 1000)

        # Calculate the total activity in the image
        self.A_image_MBq = np.sum(pet_img_act_arr)/1e6

        # Calculate scaling factor
        scaling_factor=self.A_th_MBq/self.A_image_MBq

        print(f'\t\t Theoretical activity for current timepoint = {self.A_th_MBq} MBq')
        print(f'\t\t Quantified image activity for current timepoint = {self.A_image_MBq} MBq')
        print('\t\t Applied scaling factor: ', scaling_factor)

        # Apply scaling factor to image
        pet_img_act_arr = pet_img_act_arr * scaling_factor

        #Revert back to a concentration map

        rescaled_pet_img_arr = pet_img_act_arr *(1000/pxl_vol)

        # Convert back to image

        rescaled_pet_img=sitk.GetImageFromArray(rescaled_pet_img_arr)

        # Set the origin, spacing, and direction of the image to match those of original image
        rescaled_pet_img.SetOrigin(pet_img_origin)
        rescaled_pet_img.SetSpacing(pet_img_spacing)
        rescaled_pet_img.SetDirection(pet_img_direction)

        # Export Dosemap to output folder
        filename=os.path.splitext(os.path.basename(self.image_file))[0]
        self.image_file=os.path.join(self.image_folder,f'rescaled_{filename}.mha')
        sitk.WriteImage(rescaled_pet_img,  self.image_file)

    def decay_correction(self):
        '''
        Decay correct the initial concentration value to acquisition time
        :return:
        '''

        for dcm_folder in os.listdir(args.input_image_folder):

            if 'rescaled_'+os.path.basename(dcm_folder) == os.path.basename(self.image_file.rstrip('.mha')):  # Check if the dicom folder name matches the current image

                dcm_files = os.listdir(os.path.join(args.input_image_folder, dcm_folder))

                # Read the first and last images in the dicom pile (This is because for multistep acquisition, the first image doesn't always correspond to the first step)
                first_dcm_file = pydicom.dcmread(os.path.join(args.input_image_folder, dcm_folder, dcm_files[0]))
                last_dcm_file = pydicom.dcmread(os.path.join(args.input_image_folder, dcm_folder, dcm_files[-1]))

                # Extract calibration information (Date and Time) => 0018,1074 & 0018,1078 tags
                D0 = first_dcm_file[0x0054, 0x0016][0][0x0018, 0x1078].value[:8]  # Calibration date
                T0 = first_dcm_file[0x0054, 0x0016][0][0x0018, 0x1078].value[8:14]  # Calibration time

                # Extract acquisition date and time from the first and last slice of the dicom pile, only keep the smallest acquisition start time => 0008,002 & 0008,0032 tags
                D_first = first_dcm_file.AcquisitionDate  # Acquisition date on the first image
                T_first = first_dcm_file.AcquisitionTime  # Acquisition time on the first image

                D_last = last_dcm_file.AcquisitionDate  # Acquisition date on the last image
                T_last = last_dcm_file.AcquisitionTime  # Acquisition time on the last image

                # Create datetime objects | Format YYYY,MM,DD,HH,MM,SS
                calib_datetime = datetime(int(D0[0:4]), int(D0[4:6]), int(D0[6:8]), int(T0[0:2]), int(T0[2:4]),
                                          int(T0[4:6]))
                first_datetime = datetime(int(D_first[0:4]), int(D_first[4:6]), int(D_first[6:8]), int(T_first[0:2]),
                                          int(T_first[2:4]), int(T_first[4:6]))
                last_datetime = datetime(int(D_last[0:4]), int(D_last[4:6]), int(D_last[6:8]), int(T_last[0:2]),
                                         int(T_last[2:4]), int(T_last[4:6]))

                # Compare the two times and choose the earliest one. If they are equal (ex: 1 bed acquisition) chose either one for the future
                if first_datetime < last_datetime:
                    acquisition_datetime = first_datetime

                elif first_datetime > last_datetime:
                    acquisition_datetime = last_datetime

                elif first_datetime == last_datetime:
                    acquisition_datetime = first_datetime

                # Calculate the time difference in seconds
                time_diff_sec = (acquisition_datetime - calib_datetime).total_seconds()

                # Calculate the theoretical activity in MBq at the acquisition start Date and Time
                lambda_phys = math.log(2) / self.radionuc_half_life_sec[self.radionuc]
                self.sphere_con_kBqmL = (self.sphere_con_kBqmL * math.exp(-lambda_phys * time_diff_sec))

        print(f'\t\t Theoretical sphere concentration for current timepoint = {round(self.sphere_con_kBqmL,2)} kBq/mL')

    def calculate_RC(self):

        spheres_folder_path = os.path.join(args.output, 'Binary_images', 'Hot_Spheres')

        hot_spheres = os.listdir(spheres_folder_path)

        for k in range(6):
            p1 = sub.Popen(
                ['clitkImageStatistics', '-i', self.image_file, '-m',
                 os.path.join(spheres_folder_path, hot_spheres[k]), '-v'],
                stdout=sub.PIPE)
            output1 = p1.communicate()[0].decode()  # Save the output which includes the mean value
            for row in output1.splitlines():
                if "Mean" in row:
                    mean_val = row.split(":")
                    self.C_moy_kBqmL[k] = (float(mean_val[1].strip()))/1e3  # Mean counts in hot sphere k
                if "Volume (cc)" in row:
                    vol_roi = row.split(":")
                    self.sphere_vol_cc[k] = float(vol_roi[1].strip()) # Volume of the ROI used for analysis
                if "Max" in row:
                    max_val = row.split(":")
                    max_val = max_val[1].strip()
                if "Min" in row:
                    min_val = row.split(":")
                    min_val = min_val[1].strip()

            self.RC[k]= self.C_moy_kBqmL[k]/self.sphere_con_kBqmL #Calculate RC

            #Update stat table
            new_row=pd.DataFrame([[self.imagename,hot_spheres[k].split('.')[0],self.sphere_vol_cc[k],min_val,max_val,float(mean_val[1].strip()),self.sphere_con_kBqmL,self.RC[k]]],columns=self.df_stat_image.columns)
            self.df_stat_image = pd.concat([self.df_stat_image, new_row], ignore_index=True)

        self.data['Sphere Volume (cc)']=self.sphere_vol_cc
        self.data['RC']=self.RC

class BlurredImage(PetImage):
    # Class object attributes (attributes in common for all objects of the class)
    image_folder = args.input_blurred_folder
    sbr=args.concentration_ratio
    df_stat_image=pd.DataFrame(columns=['Name','FWHM (mm)','VOI','Volume (cc)','Min','Max','Mean','RC']) #Dataframe that will store image statistics

    def __init__(self,image_file):
        self.image_file=os.path.join(self.image_folder,image_file) # PET Image File path
        self.label_image_file=args.label_blurred_file # Label file path
        self.imagename=os.path.splitext(os.path.basename(self.image_file))[0]
        self.fwhm_mm=float(self.imagename.split('FWHM')[-1].split('FWHM')[-1].split('.nii')[0])
        #Initialise attributes
        self.C_moy=np.zeros(6) # Average value in the spheres spheres extracted from the image
        self.sphere_vol_cc=np.zeros(6) # Sphere volume extracted through binary ROI
        self.RC=np.zeros(6) #Recovery coefficients will be stored here
        self.data=pd.DataFrame({
            'Sphere Diameter (mm)': [10, 13, 17, 22, 28, 37],
            'Sphere Volume (cc)': [0, 0, 0, 0, 0, 0],
            'RC': [0, 0, 0, 0, 0, 0]
        })


    def calculate_RC(self):

        spheres_folder_path = os.path.join(args.output, 'Binary_images', 'Hot_Spheres')

        hot_spheres = os.listdir(spheres_folder_path)

        for k in range(6):
            p1 = sub.Popen(
                ['clitkImageStatistics', '-i', self.image_file, '-m',
                 os.path.join(spheres_folder_path, hot_spheres[k]),'-v'],
                stdout=sub.PIPE)
            output1 = p1.communicate()[0].decode()  # Save the output which includes the mean value
            for row in output1.splitlines():
                if "Mean" in row:
                    mean_val = row.split(":")
                    self.C_moy[k] = (float(mean_val[1].strip()))  # Mean counts in hot sphere k
                if "Volume (cc)" in row:
                    vol_roi = row.split(":")
                    self.sphere_vol_cc[k] = float(vol_roi[1].strip()) # Volume of the ROI used for analysis
                if "Max" in row:
                    max_val = row.split(":")
                    max_val = max_val[1].strip()
                if "Min" in row:
                    min_val = row.split(":")
                    min_val = min_val[1].strip()

            self.RC[k]= self.C_moy[k]/self.sbr #Calculate RC

            #Update stat table
            new_row=pd.DataFrame([[self.imagename,self.fwhm_mm,hot_spheres[k].split('.')[0],self.sphere_vol_cc[k],min_val,max_val,float(mean_val[1].strip()),self.RC[k]]],columns=self.df_stat_image.columns)
            self.df_stat_image = pd.concat([self.df_stat_image, new_row], ignore_index=True)

        self.data['Sphere Volume (cc)']=self.sphere_vol_cc
        self.data['RC']=self.RC


def main():

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output):
        print('Creating output folder: ', args.output)
        os.makedirs(args.output)

    # Instantiation of new objects which will be stored in the image_object_list
    image_object_list = instantiate_PETobject()
    blurred_image_object_list= instantiate_Blurredobject()
    # Check for unmatched PET images
    if len(image_object_list) != len(os.listdir(args.input_image_folder)):
        # Store the images in the input folder and in the object list into two sets

        image_files_set = set(os.listdir(args.input_image_folder))
        image_object_set = set([])

        # Fill the image_object_set with the names of the files in the image object list
        for k in range(len(image_object_list)):
            image_object_set.add(os.path.split(image_object_list[k].image_file)[-1])

        # Identify the files that don't match between the two (images that have no corresponding label image)
        images_with_no_mask = list(image_files_set - image_object_set)

        raise Exception(f"The following PET images have no corresponding label image: {'|'.join(images_with_no_mask)}")
        # print('Warning: One or more PET images have no corresponding label image. They will not be analysed')

    if (args.dicom_flag):

        if os.path.exists(os.path.join(args.output, 'mha_images')):
            shutil.rmtree(os.path.join(args.output, 'mha_images'))

        os.makedirs(os.path.join(args.output, 'mha_images'))


    #Check the system Os to define the destination for temporary file saving
    if platform.system()=='Windows':
        dst='C:/Coupes'
    else:
        dst='/tmp/Coupes'

    df_stat_total_PET = pd.DataFrame(columns=['Name', 'VOI', 'Volume (cc)', 'Min', 'Max', 'Mean','Real con (kBq/mL)','RC'])  # Dataframe that will store image statistics
    df_stat_total_Blurred = pd.DataFrame(columns=['Name', 'FWHM (mm)', 'VOI', 'Volume (cc)', 'Min', 'Max', 'Mean','RC'])  # Dataframe that will store image statistics

    # Loop through every image object
    print('\n##########PET IMAGE ANALYSIS#########')
    for image_object in image_object_list:
        # Step 1: Check if DICOM flag is active

        if (image_object.dicom_flag):  # Convert DICOM images to mha
            print('Converting PET Image from DICOM to mha')
            image_object.dicom2Image(dst)

        print('\nAnalaysing: ', os.path.split(image_object.image_file)[-1])

        print('\tChosen Mask image: ', image_object.label_image_file)

        # Step 2: Create binary masks for the hot spheres
        print('\n\tCreating Binary Masks')
        image_object.binarize_image()

        # Step 3: Extract theoretical activity if PET image and rescale image
        if (args.auto_scale) or (args.theoretical_act_file is not None):
            print('\n\t Rescaling PET image')
            image_object.extract_theoretical_act()
            image_object.rescale_PET_image()

        # Step 4: Auto-decay the sphere concentration if necessary
        if args.dicom_flag and args.auto_decay:
            print('\n\t Auto-decaying sphere concentration')
            image_object.decay_correction()

        # Step 5: Calculate RC in spheres
        print('\n\tCalculating RC values')
        image_object.calculate_RC()

        # Update stat table
        df_stat_total_PET=pd.concat([df_stat_total_PET, image_object.df_stat_image], ignore_index=True)

        # Delete Binary_images folder from output folder
        shutil.rmtree(os.path.join(args.output, 'Binary_images'))

    print('\n##########BLURRED IMAGE ANALYSIS#########')
    for blurred_image_object in blurred_image_object_list:

        print('\nAnalaysing: ', os.path.split(blurred_image_object.image_file)[-1])

        print('\tChosen Mask image: ', blurred_image_object.label_image_file)

        # Step 1: Create binary masks for the hot spheres
        print('\n\tCreating Binary Masks')
        blurred_image_object.binarize_image()

        # Step 2: Calculate RC
        print('\n\tCalculating RC values')
        blurred_image_object.calculate_RC()

        # Update stat table
        df_stat_total_Blurred = pd.concat([df_stat_total_Blurred, blurred_image_object.df_stat_image], ignore_index=True)

        # Delete Binary_images folder from output folder
        shutil.rmtree(os.path.join(args.output, 'Binary_images'))


    #Export stat tables
    df_stat_total_PET.to_csv(os.path.join(args.output,f'PET_image_Statistics.csv'),index=False)
    df_stat_total_Blurred.to_csv(os.path.join(args.output, f'Blurred_image_Statistics.csv'), index=False)

    # Compare RC values
    df_rs_analysis_sorted=estimate_RS(image_object_list,blurred_image_object_list)

    #Plots
    plot_rc_comp(image_object_list)
    plot_rmse_vs_fwhm(df_rs_analysis_sorted)
    plot_pet_blurred_rc_comp(image_object_list,blurred_image_object_list)



if __name__ == "__main__":
    main()