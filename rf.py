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
            if time.time() - prevTime > 0.004:
                self.sendToArduino()
                prevTime = time.time()
            

class RF:

    def __init__(self, com_port, plot = False, baud_rate=9600):

        self.phi = deg2rad(np.array([0,0,0]))
        self.theta = deg2rad(np.array([0,0,0]))
        self.zeta = deg2rad(np.array([64, 53, 48.8]))

        self.phi_old = np.array([0,0,0])
        self.theta_old = np.array([0,0,0])

        self.phi_dot = np.array([0,0,0])
        self.theta_dot = np.array([0,0,0])

        self.index_l = [0.044, 0.029, 0.019]
        self.index_r = [0.03, 0.059, 0.05, 0.048, 0.016, 0.014]
        self.A = np.array([0.012, 0.07])

        self.x_e = 0
        self.y_e = 0

        self.vQ = np.array([0,0])
        self.vQ_old = np.array([0,0])

        self.phid = 0
        self.phid_old = 0

        self.mass = 1

        self.plot = plot

        self.F_FSR = 0
        self.F_finger = np.array([0,0])
        self.F_tactip = np.array([0,0])

        self.index_Jf = np.array([[0,0],[0,0]])
        self.index_Je = np.array([0,0,0])

        self.delta_t = 1

        self.debug = 0
        self.blocking = False

        self.com = serial.Serial()
        self.com.port = com_port
        self.com.baudrate = baud_rate
        self.com.timeout = 0.1
        self.com.writeTimeout=0
        self.com.open()

        self.serial_thread = Serial_Thread(self.com)
        print('Starting Serial Thread...')
        self.serial_thread.start()

        keyboard.add_hotkey('x', self.exit_key)
        keyboard.add_hotkey('b', self.block_key)

        if plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_ylim(-0.1,0.15)
            self.ax.set_xlim(-0.1,0.15)
            self.ax.set_title('Finger and RF')
            #self.ax.hold(True)
            
            self.points_rf = self.ax.plot([0]*7,[0]*7, c='r', animated=True)[0]
            self.points_f = self.ax.plot([0]*4,[0]*4, c='b', animated=True)[0]
            self.fig.show()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            time.sleep(1)


    def exit_key(self):
        # Button press that exits the program
        keyboard.press('x')
        self.blocking = False
        time.sleep(0.01)
        print('Closing all...')
        os._exit(1)

    def block_key(self):
        keyboard.press('b')
        if self.blocking == True:
            self.blocking=False
            print('Unblocking')
        elif self.blocking == False:
            self.blocking = True
            print('Blocking')


    def update_message(self, phi_d_new, F_res):
        # Update the parameters 
        message = str(phi_d_new)
        self.serial_thread.send_msg = message
        return 0

    def parse_input(self):
        # Recieve the measurements from the glove
        data_received = self.serial_thread.recv_msg
        data = data_received
        if not data == 'XXX':
            #data = data[:18].decode('utf-8')
            data = data.split(',')
            #print(data)
            if len(data)==5:
                self.phi[0] = deg2rad(float(data[0])) # set phi_1
                self.phi[1] = deg2rad(float(data[1])) # set phi_2
                self.phi[2] = deg2rad(float(data[2]))

                self.F_FSR = float(data[3])
                self.debug = data[4]

                #print(data[3]) # get the force from the FSR
            else:
                a=0
                #print(data)
                #print(self.sending_thread.message)

    def forwards_kinematics(self):
        '''
        Calculate the Location of the finger - RF connection
        '''

        zeta = self.zeta
        r = self.index_r
        A = self.A

        self.x_e = r[0]*np.cos(self.phi[0]) + r[1]*np.cos(self.phi[0]+zeta[0]) + r[2]*np.cos(self.phi[0]+self.phi[1]+zeta[0]) + r[3]*np.cos(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]) \
            +r[4]*np.cos(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]+zeta[2]) + r[5]*np.cos(self.phi[0]+self.phi[1]+self.phi[2]+zeta[0]+zeta[1]+zeta[2]) - A[0]


        self.y_e = r[0]*np.sin(self.phi[0]) + r[1]*np.sin(self.phi[0]+zeta[0]) + r[2]*np.sin(self.phi[0]+self.phi[1]+zeta[0]) + r[3]*np.sin(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]) \
            +r[4]*np.sin(self.phi[0]+self.phi[1]+zeta[0]+zeta[1]+zeta[2]) + r[5]*np.sin(self.phi[0]+self.phi[1]+self.phi[2]+zeta[0]+zeta[1]+zeta[2]) - A[1]


        return 0


    def inverse_kinematics(self):
        '''
        Use inverse kinematics to get the pose of the finger
        '''
        l = self.index_l

        phi_e = np.sum(self.phi) + np.sum(self.zeta) - np.pi/2

        x_w = self.x_e - (l[2]/2)*np.cos(phi_e)
        y_w = self.y_e - (l[2]/2)*np.sin(phi_e)

        alpha = np.arctan(y_w/x_w)

        test = (l[0]**2 + l[1]**2 - x_w**2 - y_w**2)/(2*l[0]*l[1])

        self.theta[1] = np.pi - np.arccos((l[0]**2 + l[1]**2 - x_w**2 - y_w**2)/(2*l[0]*l[1]))
        self.theta[0] = alpha - np.arccos((x_w**2 + y_w**2 + l[0]**2 - l[1]**2)/(2*l[0]*np.sqrt(x_w**2 + y_w**2)))
        self.theta[2] = (phi_e - self.theta[1] - self.theta[0])

        return 0


    def calculate_velocities(self):
        '''
        Calculate approx joint velocities
        '''
        self.theta_dot = 1/self.delta_t * (self.theta - self.theta_old)
        self.phi_dot = 1/self.delta_t * (self.phi - self.phi_old)

        self.theta_old = self.theta.flatten()
        self.phi_old = self.phi.flatten()

    
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

    def update_plot(self):

        # Calc the pose
        zeta = self.zeta
        l = self.index_l
        r = self.index_r
        A = -self.A

        theta = self.theta
        phi = self.phi
        origin = np.array([0,0])

        # Calculate the finger points
        fpoint1 = origin + np.array([l[0]*np.cos(theta[0]), l[0]*np.sin(theta[0])])
        fpoint2 = fpoint1 + np.array([l[1]*np.cos(theta[0]+theta[1]), l[1]*np.sin(theta[0]+theta[1])])
        fpoint3 = fpoint2 + np.array([l[2]*np.cos(theta[0]+theta[1]+theta[2]), l[2]*np.sin(theta[0]+theta[1]+theta[2])])
        fpoints = np.array([origin, fpoint1, fpoint2, fpoint3])

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


    def calib_fsr(self):
        print('Hold finger still, finding force rest point...')
        time.sleep(2)
        rest_list = []
        for i in range(20):
            self.parse_input()
            rest_list.append(self.F_FSR)
            time.sleep(0.01)
        self.rest_point = np.mean(rest_list)
        time.sleep(1)

    def calib_pos(self):
        print('Straighten Finger to Obtain Open Pos...')
        time.sleep(2)
        self.parse_input()
        min_point = self.theta.sum()
        print('Close Finger to Obtain Closed Pos...')
        time.sleep(2)
        self.parse_input()
        max_point = self.theta.sum()
        print('Calibration complete...')
        return min_point, max_point



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


def main():

    # init tactip
    print('Initialising TacTip...')
    finger_name = 'Index'
    #tactip = TacTip(320,240,40, finger_name, thresh_params[finger_name][0], thresh_params[finger_name][1], crops[finger_name], -1, process=True, display=True)
    #tactip.start_cap()
    time.sleep(1)
    #tactip.start_processing_display()

    # init t-mo
    #T = Model_O('/dev/ttyUSB0', 1,4,3,2,'MX', 0.4, 0.21, -0.1, 0.05)
    finger_dict ={'Thumb':3,'Middle':2,'Index':1}
    #T.reset() # reset the hand

    rf = RF('COM3', True, 115200)
    plot = True

    # Find the rest force value when no movement
    rf.calib_fsr()
    # Do another calib procedure to set max and min points for finger movement.
    min_point, max_point = rf.calib_pos()
    # Use these to scale the finger pos bewteen 0 and 1

    while True: # the main loop controlling the glove.
        time1 = time.time()
        # Get data from ard and calculate system pose
        rf.parse_input() # read from serial and update params
        rf.F_FSR = -(rf.F_FSR - rf.rest_point)
        rf.forwards_kinematics() # find the fingertip position
        rf.inverse_kinematics() # get the full pose of the system


        # TODO:get force and apply blocking if above threshold
        #tactip_force = tactip.force
        #print(tactip_force)
        #if tactip_force > 2: # modify this thresh
          #  rf.blocking = True
        #else:
          #  rf.blocking = False

        if rf.blocking:
            rf.update_message(1,0)
            if rf.FSR < 2: # modify this tresh
                rf.blocking = False
            # Do not move the T-Mo finger if blockin is enabled.
            # Eventually add code here to move slightly depending on
            # Different between fsr force and tactip force. - send phi1 deflection to rf to add o fixed_pos
        else:
            rf.update_message(0,0)
            # get t-mo pos and send to hand
            tmo_signal = rf.theta.sum()
            #scaled_signal = 
            #T.moveMotor(finger_dict[finger_name], scaled_signal)

        print(rf.debug)
        # Update the plot
        if plot:
            rf.update_plot() # Update the realtime plot
        time.sleep(0.001)
        time2 = time.time()
        rf.delta_t=time2-time1



if __name__ == '__main__':
    main()
