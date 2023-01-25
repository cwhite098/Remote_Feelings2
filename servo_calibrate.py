import serial
from sys import stdout


def main():

    port = 'COM6'
    baud_rate = 9600
    com = serial.Serial()
    com.port = port
    com.baudrate = baud_rate
    com.timeout = 0.1
    com.writeTimeout=0

    com.open()

    key_press = None

    while True:
        key_press = input('Press c to Advance Servo: ')

        if key_press:
            com.write(key_press)
            key_press = None
            stdout.flush()

    return 0



if __name__ == '__main__':
    main()