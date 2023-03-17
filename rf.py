import numpy as np
import threading
import serial
import time
import cv2 as cv
import keyboard
from hardware.hardware import TacTip
from hardware.hands import Model_O
from image_processing import crops, thresh_params
import os
import platform
if platform.system() == 'Darwin':   # fixes plots not working on mac
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sys

# Define Globals
startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False


class Serial_Thread(threading.Thread):

    def __init__(self, com):
        threading.Thread.__init__(self)

        self.com = com
        self.recv_msg = ''
        self.send_msg = ''

    def recvArduino(self):

        global dataStarted, dataBuf, messageComplete

        if self.com.inWaiting() > 0 and messageComplete == False:
            x = self.com.read().decode("utf-8") # decode needed for Python3
            
            if dataStarted == True:
                if x != endMarker:
                    dataBuf = dataBuf + x
                else:
                    dataStarted = False
                    messageComplete = True
            elif x == startMarker:
                dataBuf = ''
                dataStarted = True
        
        if (messageComplete == True):
            messageComplete = False
            return dataBuf
        else:
            return "XXX" 

    
    def sendToArduino(self):
    
        # this adds the start- and end-markers before sending
        global startMarker, endMarker
        
        stringWithMarkers = (startMarker)
        stringWithMarkers += self.send_msg
        stringWithMarkers += (endMarker)

        self.com.write(stringWithMarkers.encode('utf-8')) # encode needed for Python3


    def run(self):
        prevTime = time.time()

        while True:
            # Check for arduino msg
            received_msg = self.recvArduino()
            if not received_msg == 'XXX':
                self.recv_msg = received_msg
                pass
            # Send message at a given interval
            if time.time() - prevTime > 0.08:
                self.sendToArduino()
                prevTime = time.time()

class RF_Finger:
    def __init__(self, finger_name, plot, l, r, A):

        self.name  = finger_name

        # Params for kinematics
        self.l = l
        self.r = r
        self.A = A

        self.F_f_old = 0

        # Joint angles
        self.phi = deg2rad(np.array([0,0,0]))
        if not self.name == 'Thumb':
            self.theta = deg2rad(np.array([0,0,0]))
        else:
            self.theta = deg2rad(np.array([0,0]))
        self.zeta = deg2rad(np.array([64, 53, 48.8]))

        self.theta_old = deg2rad(np.array([0,0,0]))
        self.phi_old = deg2rad(np.array([0,0,0]))

        self.F_FSR = 0
        self.F_finger = np.array([0,0])
        self.F_tactip = np.array([0,0])
        self.fsr_restpoint = 0

        self.min_point = 0
        self.max_point = 0

        self.blocking = False
        self.deviation = 1
        self.plot = plot

        if plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_ylim(-0.1,0.15)
            self.ax.set_xlim(-0.1,0.15)
            self.ax.set_title(self.name)
            #self.ax.hold(True)
            
            self.points_rf = self.ax.plot([0]*7,[0]*7, c='r', animated=True)[0]
            self.points_f = self.ax.plot([0]*4,[0]*4, c='b', animated=True)[0]
            self.fig.show()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            time.sleep(1)


    def forwards_kinematics(self):
        '''
        Calculate the Location of the finger - RF connection
        '''

        zeta = self.zeta
        r = self.r
        A = self.A

        self.x_e = r[0]*np.cos(self.phi[0]) + r[1]*np.cos(self.phi[0]+zeta[0]) + r[2]*np.cos(self.phi[0]+self.phi[1]+zeta[0]) + r[3]*np.cos(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]) \
            +r[4]*np.cos(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]+zeta[2]) + r[5]*np.cos(self.phi[0]+self.phi[1]+self.phi[2]+zeta[0]+zeta[1]+zeta[2]) - A[0]


        self.y_e = r[0]*np.sin(self.phi[0]) + r[1]*np.sin(self.phi[0]+zeta[0]) + r[2]*np.sin(self.phi[0]+self.phi[1]+zeta[0]) + r[3]*np.sin(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]) \
            +r[4]*np.sin(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]+zeta[2]) + r[5]*np.sin(self.phi[0]+self.phi[1]+self.phi[2]+zeta[0]+zeta[1]+zeta[2]) - A[1]

    def inverse_kinematics(self):
        '''
        Use inverse kinematics to get the pose of the finger
        Will work for index or middle finger
        '''
        l = self.l
        if not self.name == 'Thumb':
            phi_e = np.sum(self.phi) + np.sum(self.zeta) - np.pi/2

            x_w = self.x_e - (l[2]/2)*np.cos(phi_e)
            y_w = self.y_e - (l[2]/2)*np.sin(phi_e)

            alpha = np.arctan(y_w/x_w)

            arg_theta1 = (l[0]**2 + l[1]**2 - x_w**2 - y_w**2)/(2*l[0]*l[1])
            arg_theta0 = (x_w**2 + y_w**2 + l[0]**2 - l[1]**2)/(2*l[0]*np.sqrt(x_w**2 + y_w**2))

            self.theta[1] = np.pi - np.arccos(np.clip(arg_theta1,-1,1))
            self.theta[0] = alpha - np.arccos(np.clip(arg_theta0,-1,1))
            self.theta[2] = (phi_e - self.theta[1] - self.theta[0])
        else:
            arg_theta2 = (self.x_e**2 + self.y_e**2 - l[0]**2 - (l[1]/2)**2)/(2*l[0]*(l[1]/2))           
            self.theta[1] = np.arccos(np.clip(arg_theta2,-1,1))
            arg_theta1_1 = self.y_e/self.x_e
            arg_theta1_2 = ((l[1]/2)*np.sin(self.theta[1]))/(l[0]+(l[1]/2)*np.cos(self.theta[1]))
            self.theta[0] = np.arctan(np.clip(arg_theta1_1,-1,1)) - np.arctan(np.clip(arg_theta1_2,-1,1))

    def get_finger_force(self, F_FSR):
        F_r = F_FSR
        xsi1 = self.phi[0] + self.zeta[0]
        xsi2 = self.phi[1] + self.zeta[1] + self.zeta[2]
        xsi3 = self.phi[2]

        alpha = np.pi - xsi1 - xsi2 - xsi3

        #F_tau = 0.1765197 / self.r[-1]
        F_tau = F_r * 0.02

        #0.9 is the mass of the arm, change this - have a self.mass
        self.F_f = np.array([-F_r,0])  + np.array([F_tau * np.cos(alpha), -F_tau*np.sin(alpha)])

        self.F_f = np.linalg.norm(self.F_f)*10 # scale because its too small
        self.F_f = 0.6*self.F_f + (1-0.6)*self.F_f_old # lpf
        self.F_f = self.F_f*np.sign(F_FSR) # use FSR to get direction
        self.F_f_old = self.F_f # save prev for filter


    def get_signal(self):

        self.signal = (self.theta.sum() - self.min_point)/(self.max_point - self.min_point) * 0.8
        

    def calculate_velocities(self, delta_t):
        '''
        Calculate approx joint velocities
        '''
        self.theta_dot = 1/delta_t * (self.theta - self.theta_old)
        self.phi_dot = 1/delta_t * (self.phi - self.phi_old)

        self.theta_old = self.theta.flatten()
        self.phi_old = self.phi.flatten()

    def update_plot(self):

        # Calc the pose
        zeta = self.zeta
        l = self.l
        r = self.r
        A = -self.A

        theta = self.theta
        phi = self.phi
        origin = np.array([0,0])

        # Calculate the finger points
        fpoint1 = origin + np.array([l[0]*np.cos(theta[0]), l[0]*np.sin(theta[0])])
        fpoint2 = fpoint1 + np.array([l[1]*np.cos(theta[0]+theta[1]), l[1]*np.sin(theta[0]+theta[1])])
        if not self.name == 'Thumb':
            fpoint3 = fpoint2 + np.array([l[2]*np.cos(theta[0]+theta[1]+theta[2]), l[2]*np.sin(theta[0]+theta[1]+theta[2])])
            fpoints = np.array([origin, fpoint1, fpoint2, fpoint3])
        else:
            fpoints = np.array([origin, fpoint1, fpoint2])

        # Calulcate the rf points
        rfpoint1 = A
        rfpoint2 = rfpoint1 + np.array([r[0]*np.cos(phi[0]), r[0]*np.sin(phi[0])])
        rfpoint3 = rfpoint2 + np.array([r[1]*np.cos(phi[0]+zeta[0]), r[1]*np.sin(phi[0]+zeta[0])])
        rfpoint4 = rfpoint3 + np.array([r[2]*np.cos(phi[0]+phi[1]+zeta[0]), r[2]*np.sin(phi[0]+phi[1]+zeta[0])])
        rfpoint5 = rfpoint4 + np.array([r[3]*np.cos(phi[0]+phi[1]+zeta[0]+zeta[1]), r[3]*np.sin(phi[0]+phi[1]+zeta[0]+zeta[1])])
        rfpoint6 = rfpoint5 + np.array([r[4]*np.cos(phi[0]+phi[1]+zeta[0]+zeta[1]+zeta[2]), r[4]*np.sin(phi[0]+phi[1]+zeta[0]+zeta[1]+zeta[2])])
        rfpoint7 = rfpoint6 + np.array([r[5]*np.cos(phi[0]+phi[1]+phi[2]+zeta[0]+zeta[1]+zeta[2]), r[5]*np.sin(phi[0]+phi[1]+phi[2]+zeta[0]+zeta[1]+zeta[2])])
        rfpoints = np.array([rfpoint1, rfpoint2, rfpoint3, rfpoint4, rfpoint5, rfpoint6, rfpoint7])

        # Update the plot
        self.fig.canvas.restore_region(self.background)

        self.points_rf.set_xdata(rfpoints[:,0])
        self.points_rf.set_ydata(-rfpoints[:,1])
        self.points_f.set_xdata(fpoints[:,0])
        self.points_f.set_ydata(-fpoints[:,1])

        self.ax.draw_artist(self.points_rf)
        self.ax.draw_artist(self.points_f)

        self.fig.canvas.blit(self.ax.bbox)
        plt.pause(0.0001)

        return 0

            

class RF:

    def __init__(self, com_port, plot = False, baud_rate=9600):


        self.index = RF_Finger('Index', plot, [0.04, 0.029, 0.019], [0.03, 0.059, 0.05, 0.048, 0.016, 0.014], np.array([0.012, 0.07]))
        self.middle = RF_Finger('Middle', plot, [0.05, 0.036, 0.025], [0.03, 0.059, 0.05, 0.048, 0.016, 0.014], np.array([0.03, 0.06]))
        self.thumb = RF_Finger('Thumb', plot, [0.038, 0.034], [0.03, 0.045, 0.026, 0.03, 0, 0.014], np.array([0.005, 0.029]))
        self.fingers = [self.index, self.middle, self.thumb]

        # old bits
        self.vQ = np.array([0,0])
        self.vQ_old = np.array([0,0])
        self.phid = 0
        self.phid_old = 0
        self.mass = 1
        self.index_Jf = np.array([[0,0],[0,0]])
        self.index_Je = np.array([0,0,0])

        self.plot = plot
        self.delta_t = 1

        self.debug = 0

        self.com = serial.Serial()
        self.com.port = com_port
        self.com.baudrate = baud_rate
        self.com.timeout = 0.1
        self.com.writeTimeout=0
        self.com.open()

        self.serial_thread = Serial_Thread(self.com)
        print('Starting Serial Thread...')
        self.serial_thread.start()

        
        keyboard.add_hotkey('1', self.index_block_key)
        keyboard.add_hotkey('2', self.middle_block_key)
        keyboard.add_hotkey('3', self.thumb_block_key)
        keyboard.add_hotkey('d', self.index_deviate_key)
        keyboard.add_hotkey('x', self.exit_key)

        


    def exit_key(self):
        # Button press that exits the program
        keyboard.press('x')
        self.index.blocking = False
        time.sleep(0.01)
        print('Closing all...')
        os._exit(1)

    def index_block_key(self):
        keyboard.press('1')
        if self.index.blocking == True:
            self.index.blocking=False
            print('Unblocking Index')
            self.index.deviation = 1
        elif self.index.blocking == False:
            self.index.blocking = True
            print('Blocking Index')

    def middle_block_key(self):
        keyboard.press('2')
        if self.middle.blocking == True:
            self.middle.blocking=False
            print('Unblocking Middle')
            self.middle.deviation = 1
        elif self.middle.blocking == False:
            self.middle.blocking = True
            print('Blocking Middle')

    def thumb_block_key(self):
        keyboard.press('3')
        if self.thumb.blocking == True:
            self.thumb.blocking=False
            print('Unblocking Thumb')
            self.thumb.deviation = 1
        elif self.thumb.blocking == False:
            self.thumb.blocking = True
            print('Blocking Thumb')

    def index_deviate_key(self):
        # Func for testing inc in deviation of servo (phi_d)
        keyboard.press('d')
        self.index.deviation += 10


    def update_message(self):
        '''
        Update the message to send to the arduino.
        If blocking, send 1 which is 0 deviation blocking
        Add later: Increase the deviation with F_res for each finger
        '''
        if self.index.blocking:
            index = self.index.deviation
        else:
            index = 0
            
        if self.middle.blocking:
            middle = self.middle.deviation
        else:
            middle = 0

        if self.thumb.blocking:
            thumb = self.thumb.deviation
        else:
            thumb = 0

        message = str(index) + ',' +str(middle) + ',' + str(thumb) + ','
        self.serial_thread.send_msg = message

    def parse_input(self):
        # Recieve the measurements from the glove
        data_received = self.serial_thread.recv_msg
        data = data_received
        if not data == 'XXX':
            #data = data[:18].decode('utf-8')
            data = data.split(',')
            #print(data)
            if len(data)==13: # if full message received
                self.index.phi[0] = deg2rad(float(data[0]))
                self.index.phi[1] = deg2rad(float(data[1]))
                self.index.phi[2] = deg2rad(float(data[2]))
                self.index.F_FSR = float(data[3])

                self.middle.phi[0] = deg2rad(float(data[4]))
                self.middle.phi[1] = deg2rad(float(data[5]))
                self.middle.phi[2] = deg2rad(float(data[6]))
                self.middle.F_FSR = float(data[7])

                self.thumb.phi[0] = deg2rad(float(data[8]))
                self.thumb.phi[1] = deg2rad(float(data[9]))
                self.thumb.phi[2] = deg2rad(float(data[10]))
                self.thumb.F_FSR = float(data[11])

                self.debug = data[12]

                #print(data)
            else:
                # Don't updateif complete message not recved
                pass

    def update_fingers(self):
        self.parse_input()
        for f in self.fingers:
            f.forwards_kinematics()
            f.inverse_kinematics()
            f.F_FSR = (f.F_FSR - f.fsr_restpoint)
            f.get_finger_force(f.F_FSR)



    def calculate_phid(self):
        '''
        Use the admittance control algo to find phi_d(t+1)
        '''
        alpha = np.pi/2 - np.sum(self.theta)
        self.F_finger = self.F_FSR * (1/np.linalg.norm(np.array([-np.cos(alpha), np.sin(alpha)])))*np.array([-np.cos(alpha), np.sin(alpha)])
        F_res = self.F_finger - self.F_tactip

        # Calculate the Jacobians
        l = self.index_l
        theta = self.theta
        J_e = (1/self.phi_dot[0]) * self.theta_dot
        J_f = np.array([[l[0]*np.sin(theta[0]) + l[1]*np.sin(theta[0]+theta[1]) + l[2]*np.sin(theta[0]+theta[1]+theta[2]), \
                         l[1]*np.sin(theta[0]+theta[1]) + l[2]*np.sin(theta[0]+theta[1]+theta[2]), \
                         l[2]*np.sin(theta[0]+theta[1]+theta[2])],\
                        [-l[0]*np.cos(theta[0]) - l[1]*np.cos(theta[0]+theta[1]) - l[2]*np.cos(theta[0]+theta[1]+theta[2]), \
                         -l[1]*np.cos(theta[0]+theta[1]) - l[2]*np.cos(theta[0]+theta[1]+theta[2]), \
                         -l[2]*np.cos(theta[0]+theta[1]+theta[2])]])

        self.vQ_old = np.matmul(J_f, self.theta_dot)
        self.vQ = ((F_res * self.delta_t)/self.mass) + self.vQ_old

        J_comb = np.matmul(J_f, J_e)
        J_comb_inv = 0.5 * np.array([1/J_comb[0], 1/J_comb[1]])

        # Update phi_d
        self.phid = self.phi[2] + (np.dot(J_comb_inv, self.vQ))*self.delta_t
        if not np.isnan(self.phid):
            self.phid_old = float(self.phid)
        
        return 0

    def get_grasp_force(self):
        # Update the F_tactip - (is 2 dimensional) from the tactile images
        return 0

    


    def calib_fsr(self):
        # Loop through each finger and get the fsr restpoint
        for f in self.fingers:
            print('Hold '+ f.name + ' finger still, finding force rest point...')
            time.sleep(2)
            rest_list = []
            for i in range(20):
                self.parse_input()
                rest_list.append(f.F_FSR)
                time.sleep(0.01)
            f.fsr_restpoint = np.mean(rest_list)
            print(f.name +' FSR Midpoint: ', f.fsr_restpoint)
            time.sleep(1)


    def calib_pos(self):
        for f in self.fingers:
            print('Straighten ' + f.name + ' Finger to Obtain Open Pos...')
            time.sleep(1)
            min_list = []
            for i in range(20):
                self.parse_input()
                f.forwards_kinematics() # find the fingertip position
                f.inverse_kinematics() 
                min_point = f.theta.sum()
                print(min_point)
                min_list.append(min_point)
                time.sleep(0.01)

            print('Close ' + f.name + ' Finger to Obtain Closed Pos...')
            time.sleep(1)
            max_list = []
            for i in range(20):
                self.parse_input()
                f.forwards_kinematics() # find the fingertip position
                f.inverse_kinematics()
                max_point = f.theta.sum()
                print(max_point)
                max_list.append(max_point)
                time.sleep(0.01)

            f.min_point = np.mean(min_list)
            f.max_point = np.mean(max_list)
            print(f.name +' Calibration complete...')




def deg2rad(deg):
    '''
    Function for converting degrees to radians
    '''
    rad = (deg/360)*2*np.pi
    return rad

def rad2deg(rad):
    '''
    Function for converting radians to degrees
    '''
    deg = (rad/(2*np.pi)) * 360
    return deg

def print_info(finger_force, tactip_force, rf_debug, finger_pos):
    '''Nice func to print out important info on just one line'''
    tac_force = tactip_force
    sys.stdout.write('\rF_finger: {}, TacTip Force: {}, RF Debug: {}, Finger Pos: {}'.format(np.round(finger_force,4), np.round(tac_force,3), rf_debug,finger_pos)) #np.round(finger_pos,3)))
    sys.stdout.flush()
    

def main():

    # init tactip
    print('Initialising TacTips...')
    finger_name = 'Index'
    #index_tactip = TacTip(320,240,40, 'Index', thresh_params['Index'][0], thresh_params['Index'][1], crops['Index'], 2, process=True, display=True)
    #middle_tactip = TacTip(320,240,40, 'Middle', thresh_params['Middle'][0], thresh_params['Middle'][1], crops['Middle'], 2, process=True, display=True)
    #index_tactip.start_cap()
    #middle_tactip.start_cap()
    #time.sleep(3)
    #index_tactip.start_processing_display()
    #middle_tactip.start_processing_display

    # init t-mo
    #T = Model_O('COM12', 1,4,3,2,'MX', 0.4, 0.21, -0.21, 0.05)
    finger_dict ={'Thumb':3,'Middle':2,'Index':1}
    #T.reset() # reset the hand
    #T.adduct(0.5)

    rf = RF('COM3', False, 115200)

    # Find the rest force value when no movement
    rf.calib_fsr()

    # Do another calib procedure to set max and min points for finger movement.xx
    rf.calib_pos()

    F_finger_old = 0

    while True: # the main loop controlling the glove.
        time1 = time.time()
        # Get data from ard and calculate system pose and forces at fingertips
        # Calculate F_res and a deviation for each finger
        # rf.index.deviation = 100
        rf.update_fingers() # update the pose and the finger force

       # tactip_force = tactip.force
       # Add reset of deviation when blocking broken below
        '''tactip_force=0
        for f in rf.fingers:
            if tactip_force > 0:
                f.blocking = True
                if f.F_FSR < -100 :
                    f.blocking = False
            elif f.F_FSR < -100:
                f.blocking = False'''

        # Loop through fingers and move t-mo
        for f in rf.fingers:
            if f.plot:
                f.update_plot()
            f.get_signal()
            if f.blocking:
                rf.update_message() # modify update message to update with blocking status of all 3 fingers
                if f.F_FSR < -100: # if finger pushing back, release block - modfify this thresold
                    f.blocking = False
                    if not np.isnan(f.signal):
                        #T.moveMotor(finger_dict[f.name], f.signal) # pretty slow
                        pass
            else:
                rf.update_message()
                # get t-mo pos and send to hand
                if not np.isnan(f.signal):
                    #T.moveMotor(finger_dict[f.name], f.signal) # pretty slow
                    pass
        
        rf.update_message()
        tactip_force=0

        print_info(rf.index.F_f, tactip_force, rf.debug, rf.index.signal)
        #print(rf.index.phi)
        time.sleep(0.001)
        time2 = time.time()
        rf.delta_t=time2-time1

if __name__ == '__main__':
    main()
