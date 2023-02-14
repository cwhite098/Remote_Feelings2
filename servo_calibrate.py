import serial
from sys import stdout
import time
import os
from rf import Serial_Thread
import keyboard

# Define Globals
startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False

def exit_key():
    # Button press that exits the program
    keyboard.press('x')
    time.sleep(0.01)
    print('Closing all...')
    os._exit(1)

def main():
    keyboard.add_hotkey('x', exit_key)
    port = 'COM6'
    baud_rate = 9600
    com = serial.Serial()
    com.port = port
    com.baudrate = baud_rate
    com.timeout = 0.1
    com.writeTimeout=0
    com.open()

    serial_thread = Serial_Thread(com)
    print('Starting Serial Thread...')
    
    recv_msg='XXX'
    while True:
        key = input('Press c')
        if key=='c':
            serial_thread.send_msg = key
            serial_thread.sendToArduino()
            time.sleep(0.5)

            while recv_msg == 'XXX':
                recv_msg = serial_thread.recvArduino()

            print(recv_msg)
            recv_msg='XXX'



if __name__ == '__main__':
    main()