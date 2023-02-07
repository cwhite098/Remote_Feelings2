from vsp.detector import CvBlobDetector, optimize_blob_detector_params
import cv2
import numpy as np
import json
import os
from skimage.metrics import structural_similarity as ssim


crops = {
    'Thumb': [100,0,215,240],
    'Middle': [120,0,220,240],
    'Index': [100,0,215,240]
}

thresh_params = {
    'Thumb': [13, -25],
    'Middle': [15, -25],
    'Index': [13, -25]
}

def save_json(dict, path):
    dict = json.dumps(dict)
    f = open(path,"w")
    f.write(dict)
    f.close()

def load_json(path):
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

    names = files # get the image names
        
    return np.array(frames), names


def get_all_ssim(frames):
    '''
    Get the SSIM between every frame and the default frame
    '''
    default = frames[0,:,:] # take the 1st frame as the default
    frames = frames[1:,:,:] # separate the remaining frames
    ssim_list = []

    for frame in frames: # get the ssim for every frame
        similarity = ssim(default, frame)
        ssim_list.append(similarity)

    return ssim_list

def get_blob_detector(finger_name, frames=None, refit=False):

    if refit:
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
    out_frames = []
    out_kpts = []

    det = get_blob_detector(finger_name, frames, refit=refit)
    for frame in frames:
        keypoints = det.detect(frame)
        kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]

        mask = np.zeros(frames[0].shape[:2], dtype="uint8")
        for kpt in kpts: # add all kpts to mask
            cv2.circle(mask, (145, 200), 100, 255, -1) #add circle to mask

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frames

def get_blob_locs(frames, finger_name, refit=False):
    '''
    Get the coordinates of all the blobs detected.
    '''
    return pts

def apply_thresholding(frames, params):
    '''
    Apply Gaussian adaptive thresholding to the tactile images
    '''
    thresh_width = params[0]
    thresh_offset = params[1] # unpack params
    out = []
    for frame in frames:
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_width, thresh_offset)
        out.append(frame)
    return np.array(out)


def main():

    frames, names = load_frames('Middle', crops['Middle'])
    print(frames.shape)

    # Testing the thresholding
    frames_thresh = apply_thresholding(frames, thresh_params['Middle'])
    '''
    for f in frames_thresh:
        cv2.imshow('Test', f)
        cv2.waitKey()
    '''

    # Testing the masking with blobs
    frames = mask_with_blobs(frames, 'Middle', refit=True)

    
if __name__ == '__main__':
    main()
