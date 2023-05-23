import pandas as pd
import numpy as np
from image_processing import load_frames, get_all_ssim, apply_thresholding, mask_with_blobs, get_blob_locs, crops, thresh_params
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 18})


def print_summary(df):
    print(df[['F_res','ssim']].describe())


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
    frames, names = load_frames(finger_name, df['Image_Name'], crops[finger_name])
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


def main2():
    df = pd.read_csv('data/teleop_data.csv')
    soft = df['Soft']
    firm = df['Hard']
    plt.hist(soft, color='r', alpha=0.6, label='Soft Item')
    plt.hist(firm, color='b', alpha=0.6, label = 'Firm Item')
    plt.xlabel('Deviation (microseconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Servo Deviations During Grasp')
    plt.show()

def fit_model(df):

    X = np.array(list(df['ssim'])).reshape(-1,1)
    y = np.array(df['F_res']).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    # Data for adding line to plot
    xs = np.linspace(np.min(df['ssim']), np.max(df['ssim']), 100)
    ys = reg.predict(xs.reshape(-1,1))
    # Predictions for mae calc
    predictions = reg.predict(np.array(df['ssim']).reshape(-1,1))
    mae = mean_absolute_error(df['F_res'], predictions)
    scaler = None
    return xs, ys, mae, reg, scaler


def plot_ssims(finger_name, axes, row):
    df_ind_t1, frames = load_data(finger_name, 't1')
    df_ind_t2, frames = load_data(finger_name, 't2')
    df_ind_t3, frames = load_data(finger_name, 't3')

    xs, ys, mae, reg, scaler = fit_model(df_ind_t1)
    print(finger_name+' t1: '+str(mae))
    axes[row][0].scatter(df_ind_t1['ssim'], df_ind_t1['F_res'], marker='+', alpha=0.7)
    axes[row][0].plot(xs, ys, c='k')
    axes[row][0].set_title(finger_name + ' Raw')
    axes[row][0].set_xlabel('SSIM'), axes[row][0].set_ylabel('F /N')
    axes[row][0].set_xlim(0.6,1), axes[row][0].set_ylim(0,3)

    xs, ys, mae, reg, scaler = fit_model(df_ind_t2)
    print(finger_name+' t2: '+str(mae))
    axes[row][1].scatter(df_ind_t2['ssim'], df_ind_t2['F_res'], marker='+', alpha=0.7)
    axes[row][1].plot(xs, ys, c='k')
    axes[row][1].set_title(finger_name+' Thresholded')
    axes[row][1].set_xlabel('SSIM'), axes[row][1].set_ylabel('F /N')
    axes[row][1].set_xlim(0.6,1), axes[row][1].set_ylim(0,3)

    xs, ys, mae, reg, scaler = fit_model(df_ind_t3)
    print(finger_name+' t3: '+str(mae))
    axes[row][2].scatter(df_ind_t3['ssim'], df_ind_t3['F_res'], marker='+', alpha=0.7)
    axes[row][2].plot(xs, ys, c='k')
    axes[row][2].set_title(finger_name+' Masked')
    axes[row][2].set_xlabel('SSIM'), axes[row][2].set_ylabel('F /N')
    axes[row][2].set_xlim(0.6,1), axes[row][2].set_ylim(0,3)

    return axes

def main():
    '''
    Loads the frames for 1 finger and 1 image type and gives ssim and F_res summary stats
    '''
    fig, axes = plt.subplots(3,3)
    axes = plot_ssims('Index', axes, 0)
    axes = plot_ssims('Middle', axes, 1)
    axes = plot_ssims('Thumb', axes, 2)
    fig.tight_layout()
    plt.show()


    

if __name__ == '__main__':
    main2()