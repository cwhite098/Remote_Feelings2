import pandas as pd
import numpy as np
from image_processing import load_frames, get_all_ssim, apply_thresholding, crops, thresh_params
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
    frames, names = load_frames(finger_name, crops[finger_name])
    ssims = get_all_ssim(frames)
    df['ssim_t1'] = ssims

    thresh_frames = apply_thresholding(frames, thresh_params[finger_name])
    ssims = get_all_ssim(thresh_frames)
    df['ssim_t2'] = ssims

    # Do the above for the masked frames

    # Load the keypoint locations


    print(df.head())
    return df



def main():
    df_middle = load_data('Middle')
    print_summary(df_middle)

    df_index = load_data('Index')
    print_summary(df_index)

    df_thumb = load_data('Thumb')
    print_summary(df_thumb)
    

if __name__ == '__main__':
    main()