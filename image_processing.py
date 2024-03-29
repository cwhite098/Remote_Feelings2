from vsp.detector import CvBlobDetector, optimize_blob_detector_params
import cv2
import numpy as np
import json
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd




crops = {
    'Thumb': [115,0,205,240],
    'Middle': [125,0,215,240],
    'Index': [118,0,208,240]
}

thresh_params = {
    'Thumb': [15, -25],
    'Middle': [15, -28],
    'Index': [15, -16]
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


def load_frames(finger_name, names, crop):
    files = os.listdir('images/'+finger_name)
    frames = []
    names = list(names)

    # Load the default frame
    img = cv2.imread('images/'+finger_name+'/default.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale
    img = crop_image(img, crop)
    frames.append(img)

    # Load the other frames
    for f in names:
        img = cv2.imread('images/'+finger_name+'/'+f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale
        img = crop_image(img, crop)
        frames.append(img)

    #names = files # get the image names
    names.insert(0,'default.jpg')
        
    return np.array(frames), names


def get_all_ssim(frames):
    '''
    Get the SSIM between every frame and the default frame
    '''
    default = frames[0,:,:] # take the 1st frame as the default
    frames = frames[1:,:,:] # separate the remaining frames
    ssim_list = []

    if (default.max() > 1):
        range = 255
    else:
        range=1

    for frame in frames: # get the ssim for every frame
        similarity = ssim(default, frame, data_range=range)
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

    det = get_blob_detector(finger_name, frames, refit=refit)
    for frame in frames:
        keypoints = det.detect(frame)
        kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]
        mask = np.zeros(frames[0].shape[:2], dtype="uint8")

        for kpt in kpts: # add all kpts to mask
            cv2.circle(mask, (int(kpt.pt[0]), int(kpt.pt[1])), int(kpt.size), 255, -1) #add circles to mask

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        masked_frame = masked_frame/255
        out_frames.append(masked_frame)

    return np.array(out_frames)


def get_blob_locs(frames, finger_name, refit=False):
    '''
    Get the coordinates of all the blobs detected.
    '''

    blob_locs = []
    det = get_blob_detector(finger_name, frames, refit=refit)
    for frame in frames:
        kpts_list=[]
        keypoints = det.detect(frame)
        for kp in keypoints:
            kpts_list.append([kp.point[0], kp.point[1], kp.size]) # include size, why not?
        blob_locs.append(np.array(kpts_list))

    return np.array(blob_locs)


def apply_thresholding(frames, params):
    '''
    Apply Gaussian adaptive thresholding to the tactile images
    '''
    thresh_width = params[0]
    thresh_offset = params[1] # unpack params
    out = []
    for frame in frames:
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_width, thresh_offset)
        frame =frame/255
        out.append(frame)
         
    return np.array(out)


def main2():
    finger_name = 'Index'
    path = 'data/' + finger_name + '.csv'
    df = pd.read_csv(path)
    names = df['Image_Name']
    t1_frames, names = load_frames(finger_name, names, crops[finger_name])
    print(t1_frames.shape)

    cv2.imshow('t1_test',t1_frames[1])
    cv2.waitKey()
    
    '''
    # Testing the thresholding
    t2_frames_thresh = apply_thresholding(t1_frames, thresh_params[finger_name])
    for i in range(3000):
        cv2.imshow('Test', t2_frames_thresh[i])
        cv2.waitKey()
    '''
    t3_frames = mask_with_blobs(t1_frames, finger_name, refit=False)
    for i in range(3000):
        cv2.imshow('Test', t3_frames[i])
        cv2.waitKey()


def main():
    
    finger_name = 'Thumb'
    path = 'data/' + finger_name + '.csv'
    df = pd.read_csv(path)
    names = df['Image_Name']
    t1_frames, names = load_frames(finger_name, names, crops[finger_name])
    print(t1_frames.shape)
    
    # Testing the thresholding
    t2_frames_thresh = apply_thresholding(t1_frames, thresh_params[finger_name])
    
    # Testing the masking with blobs
    t3_frames = mask_with_blobs(t1_frames, finger_name, refit=True)
    
    path = 'images/fig/'

    cv2.imshow('Test', t1_frames[0])
    cv2.waitKey()
    cv2.imwrite(path+'t1.png', t1_frames[0])
    
    cv2.imshow('Test', t2_frames_thresh[0])
    cv2.waitKey()
    cv2.imwrite(path+'t2.png', t2_frames_thresh[0])
    
    cv2.imshow('Test', t3_frames[0])
    cv2.waitKey()
    cv2.imwrite(path+'t3.jpg', 255*t3_frames[0])


    det = get_blob_detector(finger_name, t1_frames, refit=False)
    blob_frame = t1_frames[0].copy()
    keypoints = det.detect(blob_frame)
    kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]

    for kpt in kpts: # add all kpts to mask
        cv2.circle(blob_frame, (int(kpt.pt[0]), int(kpt.pt[1])), int(kpt.size), (0,0,255), 2) #add circles to mask

    cv2.imshow('Test', blob_frame)
    cv2.waitKey()
    cv2.imwrite(path+'blob.png', blob_frame)

   
    
    
    


    
if __name__ == '__main__':
    main2()
