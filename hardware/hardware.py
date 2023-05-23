import cv2
import time
from threading import Thread
import serial
import minimalmodbus as mm
from skimage.metrics import structural_similarity as ssim
from nn import PoseNet
import os
from data import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn import load_combined_dataset
from image_processing import crops, thresh_params, load_json, get_blob_detector
from vsp.detector import CvBlobDetector, optimize_blob_detector_params
import json

def ssim_to_force(ssim):
    force = (-8.2065*ssim) + 7.9869
    return force

class Force_Estimator:
    def __init__(self, net_type, img_type):
        # load network
        '''if net_type== 'Combined':
            X_train, X_test, X_val, y_train, y_test, y_val, self.scaler = load_combined_dataset(img_type)

        self.CNN = PoseNet(  conv_activation = 'elu',
                        dropout_rate = 0.001,
                        l1_rate = 0.0001,
                        l2_rate = 0.01,
                        learning_rate = 0.00001,
                        decay_rate = 0.000001,
                        dense_width = 128,
                        loss_func = 'mse',
                        batch_bool = False,
                        N_convs = 4,
                        N_filters = 256  
                        )
        nets = os.listdir('saved_nets')
        for net in nets:
            if net_type in net and img_type in net:
                print('saved_nets/'+net)
                self.CNN.load_net('saved_nets/'+net)
        self.CNN.create_network(240, 90, 1) # create the NN'''
        
        self.index_img = None
        self.middle_img = None
        self.thumb_img = None

        self.index_force = 0
        self.middle_force = 0
        self.thumb_force = 0

        # init tactips
        print('Initialising TacTips...')
        self.index_tactip = TacTip(320,240,40, 'Index', thresh_params['Index'][0], thresh_params['Index'][1], crops['Index'], 1, process=True, display=True)
        self.thumb_tactip = TacTip(320,240,40, 'Thumb', thresh_params['Thumb'][0], thresh_params['Thumb'][1], crops['Thumb'],0, process=True, display=True)
        self.middle_tactip = TacTip(320,240,40, 'Middle', thresh_params['Middle'][0], thresh_params['Middle'][1], crops['Middle'], 2, process=True, display=True)
        
        self.index_tactip.start_cap()
        self.middle_tactip.start_cap()
        self.thumb_tactip.start_cap()
        time.sleep(3)
        self.index_tactip.start_processing_display()
        self.middle_tactip.start_processing_display()
        self.thumb_tactip.start_processing_display()

        self.stop = False

    def run(self):
        while not self.stop:
            # Get fresh tactile images
            self.index_img = self.index_tactip.processed_img
            self.middle_img = self.middle_tactip.processed_img
            self.thumb_img = self.thumb_tactip.processed_img

            '''# Create batch of 3 images
            batch = np.array([self.index_img, self.middle_img, self.thumb_img])
            #print(batch.shape)
            if (not self.index_img is None) and (not self.middle_img is None) and (not self.thumb_img is None):
                #print('Making Prediction....')
                forces = self.CNN.predict(batch, verbose=0)
                forces = self.scaler.inverse_transform(forces)
                self.index_force = forces[0]-1
                self.middle_force = forces[1]
                self.thumb_force = forces[2]'''
            
            self.index_force = ssim_to_force(self.index_tactip.similarity)
            self.middle_force = ssim_to_force(self.middle_tactip.similarity)
            self.thumb_force = ssim_to_force(self.thumb_tactip.similarity)
           
            # Break on press of x
            key = cv2.waitKey(1)
            if key == ord('x'):
                self.stopped=True

    def start_predictions(self):
        Thread(target=self.run, args=()).start()




class TacTip:

    def __init__(self, width, height, fps, name, thresh_width, thresh_offset, crop, video_capture, process=False, display=True):
        # Init class vars
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.crop = crop
        self.display = display
        self.process = process

        # Open the camera and check its working
        self.vid = cv2.VideoCapture(video_capture, cv2.CAP_DSHOW)
        # https://stackoverflow.com/questions/56974772/usb-camera-opencv-videocapture-returns-partial-frames
        if not self.vid.isOpened():
            print("Cannot open camera " + self.name)
            exit()
        self.vid.set(cv2.CAP_PROP_FPS, self.fps)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.vid.set(cv2.CAP_PROP_BRIGHTNESS, 3)
    
        # params for Gaussian thresholding
        self.thresh_width = thresh_width
        self.thresh_offset = thresh_offset
        
        self.frame = None
        self.stopped = False
        self.processed_img = None
        self.initial_img = None

        # Let camera warm up
        time.sleep(3)

        ret, self.frame = self.vid.read()
        #cv2.imshow('test', self.frame)
        time.sleep(0.1)

        self.params = load_json('params/'+self.name+'.json')
        self.det = get_blob_detector(self.name, None, refit=False)
        self.similarity = 0
        
        
    def stream(self):
        '''
        Function that repeatedly polls the camera for a new frame
        '''
        # grab 1 frame and save it for ssim comparisons
        ret, frame = self.vid.read()
        if self.process:
            self.initial_img = self.process_frame(frame)

        # Capture frames from camera 
        while not self.stopped:
            ret, self.frame = self.vid.read()
            if not ret:
                print('No Frame Sad Face :(')
            #time.sleep(0.01)
            

    def process_and_display(self):
        '''
        Function that takes the most recent frame and applies the processing and displays the frame
        '''
        while not self.stopped:
            image = self.read()
            if self.process:
                image = self.process_frame(image)
            
            # show the frame
            if self.display:
                cv2.imshow(self.name, image)
            key = cv2.waitKey(1)

            self.processed_img = image
            self.similarity = ssim(self.initial_img, self.processed_img, data_range=1)
                        
            # Break on press of q
            if key == ord('x'):
                self.stopped=True
        self.vid.release()


    def save_image(self, path):
        image = self.read()

        # Dont process the saved images, this can always be done after
        #image = self.process_frame(image)

        cv2.imwrite(path, image)


    def start_cap(self):
        Thread(target=self.stream, args=()).start()


    def start_processing_display(self):
        Thread(target=self.process_and_display, args=()).start()
    

    def read(self):
        return self.frame
    

    def stop(self):
        self.stopped = True
    

    def process_frame(self, frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# convert to grayscale
        x0,y0,x1,y1 = self.crop
        frame = frame[y0:y1,x0:x1]
        #frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.thresh_width, self.thresh_offset)
        keypoints = self.det.detect(frame)
        kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]
        mask = np.zeros((240,90), dtype="uint8")
        for kpt in kpts: # add all kpts to mask
            cv2.circle(mask, (int(kpt.pt[0]), int(kpt.pt[1])), int(kpt.size), 255, -1) #add circles to mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = masked_frame/255

        



        # use nn to predict
        #self.force = self.CNN.predict(frame.reshape(1,frame.shape[0],frame.shape[1],1), verbose=0)
        #self.force = self.scaler.inverse_transform(self.force)

        return frame




class FSR:
    def __init__(self, port, baudrate):

        self.port = port
        self.baudrate = baudrate

        self.arduino = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)

        pass


    def read_sensor(self):

        self.arduino.flushInput()
        analogue_input = self.arduino.readline().decode().rstrip()

        return int(analogue_input)




class FTsensor:
    def __init__(self, port_name):

        self.port_name = port_name

        #Communication setup
        BAUDRATE=19200
        BYTESIZE=8
        PARITY="N"
        STOPBITS=1
        TIMEOUT=0.2
        SLAVEADDRESS=9

        ser=serial.Serial(port=self.port_name, baudrate=BAUDRATE, bytesize=BYTESIZE, parity=PARITY, stopbits=STOPBITS, timeout=TIMEOUT)

        packet = bytearray()
        sendCount=0
        while sendCount<50:
            packet.append(0xff)
            sendCount=sendCount+1
        ser.write(packet)
        ser.close()

        ####################
        #Setup minimalmodbus
        ####################
        #Communication setup
        mm.BAUDRATE=BAUDRATE
        mm.BYTESIZE=BYTESIZE
        mm.PARITY=PARITY
        mm.STOPBITS=STOPBITS
        mm.TIMEOUT=TIMEOUT

        #Create FT300 object
        self.ft300=mm.Instrument(self.port_name, slaveaddress=SLAVEADDRESS)


    def forceConverter(self, forceRegisterValue):
        """Return the force corresponding to force register value.
        
        input:
            forceRegisterValue: Value of the force register
            
        output:
            force: force corresponding to force register value in N
        """
        force=0

        forceRegisterBin=bin(forceRegisterValue)[2:]
        forceRegisterBin="0"*(16-len(forceRegisterBin))+forceRegisterBin
        if forceRegisterBin[0]=="1":
            #negative force
            force=-1*(int("1111111111111111",2)-int(forceRegisterBin,2)+1)/100
        else:
            #positive force
            force=int(forceRegisterBin,2)/100
        return force

    def torqueConverter(self, torqueRegisterValue):
        """Return the torque corresponding to torque register value.
        
        input:
            torqueRegisterValue: Value of the torque register
            
        output:
            torque: torque corresponding to force register value in N.m
        """
        torque=0

        torqueRegisterBin=bin(torqueRegisterValue)[2:]
        torqueRegisterBin="0"*(16-len(torqueRegisterBin))+torqueRegisterBin
        if torqueRegisterBin[0]=="1":
            #negative force
            torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin,2)+1)/1000
        else:
            #positive force
            torque=int(torqueRegisterBin,2)/1000
        return torque

    def set_zeros(self):
        #Read registers where are saved force and torque values.
        registers = self.ft300.read_registers(180,6)

        #Save measured values at rest. Those values are use to make the zero of the sensor.
        self.fxZero=self.forceConverter(registers[0])
        self.fyZero=self.forceConverter(registers[1])
        self.fzZero=self.forceConverter(registers[2])
        self.txZero=self.torqueConverter(registers[3])
        self.tyZero=self.torqueConverter(registers[4])
        self.tzZero=self.torqueConverter(registers[5])


    def read_forces(self,):
        #Read registers where are saved force and torque values.
        registers=self.ft300.read_registers(180,6)
        
        #Calculate measured value form register values
        fx = self.forceConverter(registers[0])-self.fxZero
        fy = self.forceConverter(registers[1])-self.fyZero
        fz = self.forceConverter(registers[2])-self.fzZero
        tx = self.torqueConverter(registers[3])-self.txZero
        ty = self.torqueConverter(registers[4])-self.tyZero
        tz = self.torqueConverter(registers[5])-self.tzZero

        return [fx, fy, fz, tx, ty, tz]










def main():

    print('Initialising TacTip...')
    finger_name = 'Index'
    tactip = TacTip(320,240,40, finger_name, thresh_params[finger_name][0], thresh_params[finger_name][1], crops[finger_name], 0, process=True, display=True)
    tactip.start_cap()
    time.sleep(3)
    tactip.start_processing_display()




if __name__ == '__main__':
    main()
