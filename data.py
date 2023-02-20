import pandas as pd
import numpy as np
from image_processing import load_frames, get_all_ssim, apply_thresholding, mask_with_blobs, get_blob_locs, crops, thresh_params
import matplotlib.pyplot as plt


def print_summary(df):
    print(df[['fz','ssim']].describe())


def load_data(finger_name, img_type):

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
    if img_type == 't1':
        ssims = get_all_ssim(frames)
        df['ssim'] = ssims

    if img_type == 't2':
        # Apply thresholding and get ssim for t2 processed images
        frames = apply_thresholding(frames, thresh_params[finger_name])
        ssims = get_all_ssim(frames)
        df['ssim'] = ssims

    if img_type == 't3':
        # Do the above for the masked frames
        frames = mask_with_blobs(frames, finger_name, refit=False)
        ssims = get_all_ssim(frames)
        df['ssim'] = ssims
    
    # Load the keypoint locations
    if img_type == 't4':
        frames = get_blob_locs(frames, finger_name, refit=False)

    print('Finger Name: '+finger_name)
    print('Image Type: '+img_type)
    print(df.head())

    return df, frames



def main():
    df, frames = load_data('Index', 't2')
    print_summary(df)
    print(np.max(frames[0]))
    plt.scatter(df['ssim'], df['F_res'])
    plt.show()
    

if __name__ == '__main__':
    main()