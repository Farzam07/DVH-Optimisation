# ==================================I M P O R T S=============================================================
import os
import subprocess as sub
import shutil
import argparse
import platform
from natsort import natsorted
import pandas as pd

# =============================================P A R S E R=====================================================

parser = argparse.ArgumentParser(description='Segment NEMA phantom to generate a Labelled image')
parser.add_argument('-i', '--input_image',type=str,help='Path towards the input image (Dicom folder or Single image)')
parser.add_argument('-m','--mask_image_coordinates',type=str,help='Path towards the Excel file containing the coordinates for generating the labelled images')
parser.add_argument('-dc', '--dicom_flag', help='Include this option if the static PET is in DICOM format',
                    action='store_true')
parser.add_argument('-n', '--bottom_sphere', type=str, help='name sphere bottom sphere of the IEC phantom (s1,s2,s3,...)')
parser.add_argument('-r', '--reverse', help='Include this option if phantom is reversed',action='store_true')
parser.add_argument('-spheres2D', '--spheres2D', help='Include this option if you want 2D sphere ROIs',action='store_true')
parser.add_argument('-o', '--output', type=str, help='Path to output folder')

args = parser.parse_args()
# ==================================F U N C T I O N S=============================================================================

def dicom2Image(dst,in_filepath):
    if os.path.exists(dst):
        shutil.rmtree(os.path.join(dst))

    shutil.move(in_filepath,dst)

    dirlist = []  # We create an empty list to store the path to all the DICOM files in a series
    [dirlist.append(os.path.abspath(os.path.join(dst, name))) for name in
     os.listdir(dst)]  # We store the path to all the Dicom files (files ending with dcm)
    dirlist = natsorted(dirlist)  # We sort the list by alphabetical/numerical order

    command = ' '.join(
        dirlist)  # Extract the contents of the dirlist as one string to be used in the clitkDicom2Image function

    output_file_name = os.path.split(args.input_image)[-1] + '.mha'
    output_file_path = os.path.join(os.path.split(os.path.abspath(args.input_image))[-2], 'mha_images',
                                    output_file_name)

    sub.run(f'clitkDicom2Image {command} -o {output_file_path} -t 0.001 -p',
            shell=True)  # Using a clitk function to perform the Dicom to Image transformation

    shutil.move(dst, in_filepath)  # We move the DICOM series to its original place once we are done

    return output_file_path

def generate_masks(input_image):
    #Read csv file
    coordinates_df=pd.read_csv(args.mask_image_coordinates)
    # coordinates_df=coordinates_df.fillna('')

    #Go through each row and extract coordinates
    for row in coordinates_df.iterrows():
        filename=str(row[1].loc['Name'])+'.mha'

        S1 = str(row[1].loc['S1'])
        S2 = str(row[1].loc['S2'])
        S3 = str(row[1].loc['S3'])
        S4 = str(row[1].loc['S4'])
        S5 = str(row[1].loc['S5'])
        S6 = str(row[1].loc['S6'])

        Centre = str(row[1].loc['Lung'])

        Zmin= str(row[1].loc['Zmin'])
        Zmax = str(row[1].loc['Zmax'])

        BG1 = str(row[1].loc['BG1'])
        BG2 = str(row[1].loc['BG2'])
        BG3 = str(row[1].loc['BG3'])
        BG4 = str(row[1].loc['BG4'])
        BG5 = str(row[1].loc['BG5'])
        BG6 = str(row[1].loc['BG6'])
        BG7 = str(row[1].loc['BG7'])
        BG8 = str(row[1].loc['BG8'])
        BG9 = str(row[1].loc['BG9'])
        BG10 = str(row[1].loc['BG10'])
        BG11 = str(row[1].loc['BG11'])
        BG12 = str(row[1].loc['BG12'])

        S_bottom= str(row[1].loc[args.bottom_sphere.upper()]) #Coordinates of bottom sphere (given in the -n argument)

        if (args.reverse):
            reverse_flag = '-r'
        else:
            reverse_flag = ''

        if(args.spheres2D):
            print('2D sphere option ACTIVATED')
            sphere_2D_flag='--spheres2D'
        else:
            sphere_2D_flag=''

        print(f'***************Generating {filename} ****************')
        #
        list_args_clitkExtractNEMAPhantom =['clitkExtractNEMAPhantom',
                                            '-i', input_image,
                                            '--zmin', Zmin,
                                            '--zmax', Zmax,
                                            '-o', os.path.join(args.output, filename),
                                            '--s1', S1,
                                            '--s2', S2,
                                            '--s3', S3,
                                            '--s4', S4,
                                            '--s5', S5,
                                            '--s6', S6,
                                            '--centre', Centre,
                                            '-n', args.bottom_sphere,
                                            '-s', S_bottom,
                                            '--bg1', BG1,
                                            '--bg2', BG2,
                                            '--bg3', BG3,
                                            '--bg4', BG4,
                                            '--bg5', BG5,
                                            '--bg6', BG6,
                                            '--bg7', BG7,
                                            '--bg8', BG8,
                                            '--bg9', BG9,
                                            '--bg10', BG10,
                                            '--bg11', BG11,
                                            '--bg12', BG12,
                                            reverse_flag,
                                            sphere_2D_flag]

        sub.run(list_args_clitkExtractNEMAPhantom )

# ==================================M A I N=====================================================================

def main():


    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if (args.dicom_flag):
        if os.path.exists(os.path.join(os.path.split(os.path.abspath(args.input_image))[-2], 'mha_images')):
            shutil.rmtree(os.path.join(os.path.split(os.path.abspath(args.input_image))[-2], 'mha_images'))

        os.makedirs(os.path.join(os.path.split(os.path.abspath(args.input_image))[-2], 'mha_images'))

        if platform.system() == 'Windows':
            dst = 'C:/Coupes'
        else:
            dst = '/tmp/Coupes'

        input_image=dicom2Image(dst,args.input_image)

        print('Converting PET Image from DICOM to mha')
    else:
        input_image=args.input_image

    generate_masks(input_image)



if __name__ == "__main__":
    main()
