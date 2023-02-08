import pandas as pd
import numpy as np
from image_processing import load_frames, get_all_ssim, apply_thresholding, mask_with_blobs, get_blob_locs, crops, thresh_params
import matplotlib.pyplot as plt


def print_summary(df):
    print(df[['fz','ssim_t1', 'ssim_t2', 'ssim_t3']].describe())


def load_data(finger_name):

    # Load the data
    path = 'data/' + finger_name + '.csv'
    df = pd.read_csv(path)
    df = df[['Image_Name', 'Finger_Pos',  'fx',  'fy',  'fz',  'tx',  'ty',  'tz']]

    # Modify force data
    F_res = np.sqrt(df['fx']**2 + df['fy']**2 + df['fz']**2) # calculate the magnitude of the force
    df['fz'] = -df['fz'] # reverse direction of z component
    df['F_res'] = F_res

    # Load the images and get ssim
    t1_frames, names = load_frames(finger_name, crops[finger_name])
    ssims = get_all_ssim(t1_frames)
    df['ssim_t1'] = ssims

    # Apply thresholding and get ssim for t2 processed images
    t2_frames = apply_thresholding(t1_frames, thresh_params[finger_name])
    ssims = get_all_ssim(t2_frames)
    df['ssim_t2'] = ssims

    # Do the above for the masked frames
    t3_frames = mask_with_blobs(t1_frames, finger_name, refit=False)
    ssims = get_all_ssim(t3_frames)
    df['ssim_t3'] = ssims
    
    # Load the keypoint locations
    blob_locs = get_blob_locs(t1_frames, finger_name, refit=False)

    print('Finger Name: '+finger_name)
    print(df.head())

    return df, t1_frames, t2_frames, t3_frames, blob_locs



def main():
    df_mid, t1_mid, t2_mid, t3_mid, blob_locs_mid = load_data('Middle')
    print_summary(df_mid)

    df_ind, t1_ind, t2_ind, t3_ind, blob_locs_ind = load_data('Index')
    print_summary(df_ind)

    df_thu, t1_thu, t2_thu, t3_thu, blob_locs_thu = load_data('Thumb')
    print_summary(df_thu)
    

if __name__ == '__main__':
    main()