import pandas as pd
import numpy as np
import os
import argparse
import os
import matplotlib.pyplot as plt

# =============================================P A R S E R=====================================================
parser = argparse.ArgumentParser(description='Identify the optimal reconstruction based on a root mean square error evaluation')

parser.add_argument('--rmse_folder', type=str, help='Path towards folder containing the RMSE csv files')
parser.add_argument('--small_sphere_diameter', type=int, help='Enter the diameter of the smallest visible sphere in mm')
parser.add_argument('-o', '--output', type=str, help='Path to output folder')

args = parser.parse_args()

# =============================================M A I N=========================================================
iec_spheres={
    '1':10,
    '2':13,
    '3':17,
    '4':22,
    '5':28,
    '6':37
    }

small_sphere=(list(iec_spheres.keys())[list(iec_spheres.values()).index(args.small_sphere_diameter)]) #Extract the sphere number corresponding to the

if not os.path.exists(args.output):
    os.makedirs(args.output)

# Read RMSE Tables and put everything into a single dataframe
df=pd.DataFrame()
df_nrmse_LDM=pd.DataFrame()
df_nrmse_DVK=pd.DataFrame()

df_rmse_LDM=pd.DataFrame()
df_rmse_DVK=pd.DataFrame()

for rmse_table in os.listdir(args.rmse_folder):

    if '.csv' in rmse_table and 'nRMSE' in rmse_table:
        df=pd.read_csv(os.path.join(args.rmse_folder,rmse_table))

        if 'LDM' in rmse_table:
            df_nrmse_LDM=pd.concat([df_nrmse_LDM,df])

        elif 'DVK' in rmse_table:
            df_nrmse_DVK=pd.concat([df_nrmse_DVK,df])

    elif '.csv' in rmse_table and 'RMSE' in rmse_table and 'nRMSE' not in rmse_table:
        df = pd.read_csv(os.path.join(args.rmse_folder, rmse_table))

        if 'LDM' in rmse_table:
            df_rmse_LDM = pd.concat([df_rmse_LDM, df])

        elif 'DVK' in rmse_table:
            df_rmse_DVK = pd.concat([df_rmse_DVK, df])

# Compute max-normalized RMSE values
#Small sphere
df_rmse_LDM[f'RMSE/max(S{small_sphere})']= df_rmse_LDM[f'RMSE Sphere{small_sphere}']/df_rmse_LDM[f'RMSE Sphere{small_sphere}'].max()
df_rmse_DVK[f'RMSE/max(S{small_sphere})']= df_rmse_DVK[f'RMSE Sphere{small_sphere}']/df_rmse_DVK[f'RMSE Sphere{small_sphere}'].max()

#Sphere 6 (37mm)
df_rmse_LDM['RMSE/max(S6)']= df_rmse_LDM['RMSE Sphere6']/df_rmse_LDM['RMSE Sphere6'].max()
df_rmse_DVK['RMSE/max(S6)']= df_rmse_DVK['RMSE Sphere6']/df_rmse_DVK['RMSE Sphere6'].max()

#Background
df_rmse_LDM['RMSE/max(BG)']= df_rmse_LDM['RMSE BG']/df_rmse_LDM['RMSE BG'].max()
df_rmse_DVK['RMSE/max(BG)']= df_rmse_DVK['RMSE BG']/df_rmse_DVK['RMSE BG'].max()


# Compute the Euclidean between each recon and the ideal point [0,0]
# Small sphere vs Background
#Mean dose normalized RMSE
df_nrmse_LDM[f'Euclidean distance S{small_sphere}/BG'] = np.sqrt((df_nrmse_LDM['nRMSE BG'])**2 + (df_nrmse_LDM[f'nRMSE Sphere{small_sphere}'])**2)
df_nrmse_DVK[f'Euclidean distance S{small_sphere}/BG'] = np.sqrt((df_nrmse_DVK['nRMSE BG'])**2 + (df_nrmse_DVK[f'nRMSE Sphere{small_sphere}'])**2)

# Max-normalized RMSE
df_rmse_LDM[f'Euclidean distance S{small_sphere}/BG']=np.sqrt((df_rmse_LDM['RMSE/max(BG)'])**2 + (df_rmse_LDM[f'RMSE/max(S{small_sphere})'])**2)
df_rmse_DVK[f'Euclidean distance S{small_sphere}/BG']=np.sqrt((df_rmse_DVK['RMSE/max(BG)'])**2 + (df_rmse_DVK[f'RMSE/max(S{small_sphere})'])**2)

# Sphere 6 (37mm) vs Background
# Mean dose normalized RMSE
df_nrmse_LDM['Euclidean distance S6/BG'] = np.sqrt((df_nrmse_LDM['nRMSE BG'])**2 + (df_nrmse_LDM['nRMSE Sphere6'])**2)
df_nrmse_DVK['Euclidean distance S6/BG'] = np.sqrt((df_nrmse_DVK['nRMSE BG'])**2 + (df_nrmse_DVK['nRMSE Sphere6'])**2)
# Max-normalized RMSE
df_rmse_LDM['Euclidean distance S6/BG']=np.sqrt((df_rmse_LDM['RMSE/max(BG)'])**2 + (df_rmse_LDM['RMSE/max(S6)'])**2)
df_rmse_DVK['Euclidean distance S6/BG']=np.sqrt((df_rmse_DVK['RMSE/max(BG)'])**2 + (df_rmse_DVK['RMSE/max(S6)'])**2)


#Plot nRMSE Sphere vs Background curves
fig, ax = plt.subplots(1, 2)

for k in range(len(df_nrmse_LDM)):
    ax[0].plot(df_nrmse_LDM.iloc[k]['nRMSE BG'],df_nrmse_LDM.iloc[k][f'nRMSE Sphere{small_sphere}'],'o',markersize=2,label='')

ax[0].set_title(f'LDM - $S_{{\phi={str(args.small_sphere_diameter)}mm}}$ vs BG')
ax[0].set_xlabel('nRMSE$_{BG}$')
ax[0].set_ylabel(f'nRMSE$_{{\phi={str(args.small_sphere_diameter)}mm}}$')
ax[0].grid(True)
# ax[0].set_aspect('equal', adjustable='box')

for k in range(len(df_nrmse_LDM)):
    ax[1].plot(df_nrmse_LDM.iloc[k]['nRMSE BG'],df_nrmse_LDM.iloc[k]['nRMSE Sphere6'],'o',markersize=2,label=df_nrmse_LDM.iloc[k]['Name'])

ax[1].set_title('LDM - $S_{\phi=37mm}$ vs BG')
ax[1].set_xlabel('nRMSE$_{BG}$')
ax[1].set_ylabel('nRMSE$_{\phi=37mm}$')
ax[1].grid(True)
# ax[1].set_aspect('equal', adjustable='box')

# for a in ax.flatten():
#     a.set_xlim(0, 1)
#     a.set_ylim(0, 1)

# fig.legend(loc='lower center',bbox_to_anchor=(0.5,-0.08*(len(df_nrmse_LDM)/6)),ncol=2)

plt.tight_layout()

fig2, ax2 = plt.subplots(1, 2)


for k in range(len(df_nrmse_DVK)):
    ax2[0].plot(df_nrmse_DVK.iloc[k]['nRMSE BG'],df_nrmse_DVK.iloc[k][f'nRMSE Sphere{small_sphere}'],'o',markersize=2,label='')

ax2[0].set_title(f'DVK - $S_{{\phi={str(args.small_sphere_diameter)}mm}}$ vs BG')
ax2[0].set_xlabel('nRMSE$_{BG}$')
ax2[0].set_ylabel(f'nRMSE$_{{\phi={str(args.small_sphere_diameter)}mm}}$')
ax2[0].grid(True)
# ax2[0].set_aspect('equal', adjustable='box')

for k in range(len(df_nrmse_DVK)):
    ax2[1].plot(df_nrmse_DVK.iloc[k]['nRMSE BG'],df_nrmse_DVK.iloc[k]['nRMSE Sphere6'],'o',markersize=2,label=df_nrmse_DVK.iloc[k]['Name'])

ax2[1].set_title('DVK - $S_{\phi=37mm}$ vs BG')
ax2[1].set_xlabel('nRMSE$_{BG}$')
ax2[1].set_ylabel('nRMSE$_{\phi=37mm}$')
ax2[1].grid(True)
# ax2[1].set_aspect('equal', adjustable='box')

# for a in ax2.flatten():
#     a.set_xlim(0, 1)
#     a.set_ylim(0, 1)

# fig2.legend(loc='lower center',bbox_to_anchor=(0.5,-0.08*(len(df_rmse_DVK)/6)),ncol=2)
plt.tight_layout()

fig.savefig(os.path.join(args.output,'nRMSE_Sphere_vs_BG_LDM.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(args.output,'nRMSE_Sphere_vs_BG_DVK.pdf'), bbox_inches='tight')



#############################EDIT######################################
#Plot Max normalized RMSE Sphere vs Background curves
fig3, ax = plt.subplots(1, 2)

for k in range(len(df_rmse_LDM)):
    ax[0].plot(df_rmse_LDM.iloc[k]['RMSE/max(BG)'],df_rmse_LDM.iloc[k][f'RMSE/max(S{small_sphere})'],'o',markersize=2,label='')

ax[0].set_title(f'LDM - $S_{{\phi={str(args.small_sphere_diameter)}mm}}$ vs BG')
ax[0].set_xlabel('RMSE$_{BG}$/max$_{RMSE-BG}$')
ax[0].set_ylabel(f'RMSE$_{{\phi={str(args.small_sphere_diameter)}mm}}$/max$_{{RMSE-\phi={str(args.small_sphere_diameter)}mm}}$')
ax[0].grid(True)
ax[0].set_aspect('equal', adjustable='box')

for k in range(len(df_rmse_LDM)):
    ax[1].plot(df_rmse_LDM.iloc[k]['RMSE/max(BG)'],df_rmse_LDM.iloc[k]['RMSE/max(S6)'],'o',markersize=2,label=df_rmse_LDM.iloc[k]['Name'])

ax[1].set_title('LDM - $S_{\phi=37mm}$ vs BG')
ax[1].set_xlabel('RMSE$_{BG}$/max$_{RMSE-BG}$')
ax[1].set_ylabel('RMSE$_{\phi=37mm}$/max$_{RMSE-\phi=37mm}$')
ax[1].grid(True)
ax[1].set_aspect('equal', adjustable='box')

for a in ax.flatten():
    a.set_xlim(0, 1.1)
    a.set_ylim(0, 1.1)

# fig3.legend(loc='lower center',bbox_to_anchor=(0.5,-0.08*(len(df_rmse_LDM)/6)),ncol=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

fig4, ax2 = plt.subplots(1, 2)


for k in range(len(df_rmse_DVK)):
    ax2[0].plot(df_rmse_DVK.iloc[k]['RMSE/max(BG)'],df_rmse_DVK.iloc[k][f'RMSE/max(S{small_sphere})'],'o',markersize=2,label='')

ax2[0].set_title(f'DVK - $S_{{\phi={str(args.small_sphere_diameter)}mm}}$ vs BG')
ax2[0].set_xlabel('RMSE$_{BG}$/max$_{RMSE-BG}$')
ax2[0].set_ylabel(f'RMSE$_{{\phi={str(args.small_sphere_diameter)}mm}}$/max$_{{RMSE-\phi={str(args.small_sphere_diameter)}mm}}$')
ax2[0].grid(True)
ax2[0].set_aspect('equal', adjustable='box')

for k in range(len(df_rmse_DVK)):
    ax2[1].plot(df_rmse_DVK.iloc[k]['RMSE/max(BG)'],df_rmse_DVK.iloc[k]['RMSE/max(S6)'],'o',markersize=2,label=df_rmse_DVK.iloc[k]['Name'])

ax2[1].set_title('DVK - $S_{\phi=37mm}$ vs BG')
ax2[1].set_xlabel('RMSE$_{BG}$/max$_{RMSE-BG}$')
ax2[1].set_ylabel('RMSE$_{\phi=37mm}$/max$_{RMSE-\phi=37mm}$')
ax2[1].grid(True)
ax2[1].set_aspect('equal', adjustable='box')

for a in ax2.flatten():
    a.set_xlim(0, 1.1)
    a.set_ylim(0, 1.1)

# fig4.legend(loc='lower center',bbox_to_anchor=(0.5,-0.08*(len(df_rmse_DVK)/6)),ncol=2)
plt.tight_layout()

fig3.savefig(os.path.join(args.output,'maxNormRMSE_Sphere_vs_BG_LDM.pdf'), bbox_inches='tight')
fig4.savefig(os.path.join(args.output,'maxNormRMSE_Sphere_vs_BG_DVK.pdf'), bbox_inches='tight')

############################EDIT#######################################

#Output
df_nrmse_LDM.to_csv(os.path.join(args.output, f'meanDoseNorm_Euc_distance_LDM_{os.path.basename(os.path.abspath(args.rmse_folder))}.csv'),index=False)
df_nrmse_DVK.to_csv(os.path.join(args.output, f'meanDoseNorm_Euc_distance_DVK_{os.path.basename(os.path.abspath(args.rmse_folder))}.csv'),index=False)

df_rmse_LDM.to_csv(os.path.join(args.output, f'maxNorm_Euc_distance_LDM_{os.path.basename(os.path.abspath(args.rmse_folder))}.csv'),index=False)
df_rmse_DVK.to_csv(os.path.join(args.output, f'maxNorm_Euc_distance_DVK_{os.path.basename(os.path.abspath(args.rmse_folder))}.csv'),index=False)

#Extract optimal recons

df_nrmse_LDM.set_index('Name', inplace=True) #Set Name column to index to facilitate the extraction of the optimal reconstruction
df_nrmse_DVK.set_index('Name', inplace=True) #Set Name column to index to facilitate the extraction of the optimal reconstruction

df_rmse_LDM.set_index('Name', inplace=True) #Set Name column to index to facilitate the extraction of the optimal reconstruction
df_rmse_DVK.set_index('Name', inplace=True) #Set Name column to index to facilitate the extraction of the optimal reconstruction


# Export to txt file
with open(os.path.join(args.output,'results.txt'), 'w') as file:

    # Redirect the print statements to the file
    print('Mean Dose normalized RMSE ', file=file)
    print('*'*80, file=file)
    print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - LDM :', df_nrmse_LDM[f'Euclidean distance S{small_sphere}/BG'].idxmin(), file=file)
    print('Optimal recon for Sphere 6 (37mm) - LDM :', df_nrmse_LDM['Euclidean distance S6/BG'].idxmin(), file=file)

    print('*'*80, file=file)
    print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - DVK :', df_nrmse_DVK[f'Euclidean distance S{small_sphere}/BG'].idxmin(), file=file)
    print('Optimal recon for Sphere 6 (37mm) - DVK :', df_nrmse_DVK['Euclidean distance S6/BG'].idxmin(), file=file)

    print('#'*80, file=file)
    print('Max normalized RMSE ', file=file)

    print('*'*80, file=file)
    print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - LDM :', df_rmse_LDM[f'Euclidean distance S{small_sphere}/BG'].idxmin(), file=file)
    print('Optimal recon for Sphere 6 (37mm) - LDM :', df_rmse_LDM['Euclidean distance S6/BG'].idxmin(), file=file)

    print('*'*80, file=file)
    print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - DVK :', df_rmse_DVK[f'Euclidean distance S{small_sphere}/BG'].idxmin(), file=file)
    print('Optimal recon for Sphere 6 (37mm) - DVK :', df_rmse_DVK['Euclidean distance S6/BG'].idxmin(), file=file)






print('Mean Dose normalized RMSE ')

print('*'*80)
print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - LDM :', df_nrmse_LDM[f'Euclidean distance S{small_sphere}/BG'].idxmin())
print('Optimal recon for Sphere 6 (37mm) - LDM :', df_nrmse_LDM['Euclidean distance S6/BG'].idxmin())

print('*'*80)
print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - DVK :', df_nrmse_DVK[f'Euclidean distance S{small_sphere}/BG'].idxmin())
print('Optimal recon for Sphere 6 (37mm) - DVK :', df_nrmse_DVK['Euclidean distance S6/BG'].idxmin())

print('#'*80)
print('Max normalized RMSE ')

print('*'*80)
print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - LDM :', df_rmse_LDM[f'Euclidean distance S{small_sphere}/BG'].idxmin())
print('Optimal recon for Sphere 6 (37mm) - LDM :', df_rmse_LDM['Euclidean distance S6/BG'].idxmin())

print('*'*80)
print(f'Optimal recon for Sphere {small_sphere} ({str(args.small_sphere_diameter)}mm) - DVK :', df_rmse_DVK[f'Euclidean distance S{small_sphere}/BG'].idxmin())
print('Optimal recon for Sphere 6 (37mm) - DVK :', df_rmse_DVK['Euclidean distance S6/BG'].idxmin())





