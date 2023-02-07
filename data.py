import pandas as pd
import numpy as np
from image_processing import load_frames, get_all_ssim, middle_crop
import matplotlib.pyplot as plt


'''
Need to:
    - Load csv and remove redundant cols
    - add ssim col to df
    - calc resultant force
    - check reference frame
    
'''


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
    frames, names = load_frames(finger_name, middle_crop)
    ssims = get_all_ssim(frames)
    df['ssim_t1'] = ssims

    print(df.head())
    return df



def main():
    df = load_data('Middle')
    plt.scatter(df['ssim'], df['fz'])
    plt.show()
    
if __name__ == '__main__':
    main()