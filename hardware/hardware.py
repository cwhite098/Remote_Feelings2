import cv2
import time
from threading import Thread
import serial
import minimalmodbus as mm
from skimage.metrics import structural_similarity
from nn import PoseNet
import os
from data import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        self.vid = cv2.VideoCapture(video_capture)
        if not self.vid.isOpened():
            print("Cannot open camera " + self.name)
            exit()
        self.vid.set(cv2.CAP_PROP_FPS, self.fps)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
        # params for Gaussian thresholding
        self.thresh_width = thresh_width
        self.thresh_offset = thresh_offset
        
        self.frame = None
        self.stopped = False

        # Init a network
        if self.process:
            self.CNN = PoseNet(  conv_activation = 'elu',
                        dropout_rate = 0.001,
                        l1_rate = 0.0001,
                        l2_rate = 0.01,
                        learning_rate = 0.00001,
                        decay_rate = 0.000001,
                        dense_width = 16,
                        loss_func = 'mse',
                        batch_bool = False,
                        N_convs = 4,
                        N_filters = 512
                        )
            # Load the network params
            label_type = 'fz'
            img_type = 't2'
            nets = os.listdir('saved_nets/'+ label_type)
            for net in nets:
                if name in net and img_type in net:
                    print('saved_nets/'+ label_type+'/'+net)
                    self.CNN.load_net('saved_nets/'+ label_type+'/'+net)
            
            print('Loading Data...')
            df, t2 = load_data(name, 't2')
            images = t2[1:]
            X_train, X_test, y_train, y_test = train_test_split(images, df, test_size=0.2, random_state=42)
            y_train = np.array(y_train['fz']).reshape(-1, 1) 
            # Normalise the labels
            self.scaler = StandardScaler() # fit a scaler to inverse the predictions when generated
            self.scaler.fit(y_train)
        self.force = 0

        # Let camera warm up
        time.sleep(3)

        ret, self.frame = self.vid.read()
        #cv2.imshow('test', self.frame)
        time.sleep(0.1)
        
        
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
            time.sleep(0.01)
            

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
                        
            # Break on press of q
            if key == ord('q'):
                self.stopped=True


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
        frame = cv2.adaptiveThreshold(frame, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.thresh_width, self.thresh_offset)

        # use nn to predict
        self.force = self.CNN.predict(frame.reshape(1,frame.shape[0],frame.shape[1],1), verbose=0)
        #self.force = self.scaler.inverse_transform(self.force)

        return frame

    def get_ssim(self):
        new_img = self.process_frame(self.read())

        ssim = structural_similarity(self.initial_img, new_img)

        return ssim



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
