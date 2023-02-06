from vsp.detector import CvBlobDetector, optimize_blob_detector_params
import cv2
import numpy as np
import json

def save_json(dict, path):
    json = json.dumps(dict)
    f = open(path,"w")
    f.write(json)
    f.close()

def load_josn(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def crop_image(image, crop):
    x0,y0,x1,y1 = crop
    frame = image[y0:y1,x0:x1]
    return frame

def load_frames(finger_name, crop):
    files = os.listdir('images/'+finger_name)
    frames = []
    for f in files:
        img = cv2.imread('images/'+finger_name+'/'+f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale
        img = crop_image(img, crop)
        frames.append(img)
        
    return np.array(frames)

def get_all_ssim(frames):
    '''
    Get the SSIM between every frame and the default frame
    '''
    return ssim_list

def get_blob_detector(finger_name, frames=None, refit=False):

    if refit == 'True':
        # Re-optimise the blob detector params
        params = optimize_blob_detector_params(frames,
                                           target_blobs=30,
                                           min_threshold_range=(0, 300),
                                           max_threshold_range=(0, 300),
                                           min_area_range=(0, 200),
                                           max_area_range=(0, 200),
                                           min_circularity_range=(0.1, 0.9),
                                           min_inertia_ratio_range=(0.1, 0.9),
                                           min_convexity_range=(0.1, 0.9),
                                           )
        save_json(params, 'params/'+finger_name+'.json')
        detector = CvBlobDetector(**params)


    else:
        # Load blob params for the correct finger
        params = load_json('params/'+finger_name+'.json')
        detector = CvBlobDetector(**params)

    return detector

def mask_with_blobs(frames, finger_name, refit=False):
    '''
    Obtain blobs using the detector and use them to apply a mask to
    the tactile images - blocking out background
    '''
    return frames

def get_blob_locs(frames, finger_name, refit=False):
    '''
    Get the coordinates of all the blobs detected.
    '''
    return pts

def apply_thresholding(frames):
    '''
    Apply Gaussian adaptive thresholding to the tactile images
    '''
    return frames