#include <Servo.h>
#include "serial_comms.h" //add serial class
#include "finger.h" // add finger class
#include <PID_v1.h>

// Index Finger Pins
# define INDEX_FSR_PIN A2
# define INDEX_SERV_PWM 11
# define INDEX_PHI1 A11
# define INDEX_PHI2 A4
# define INDEX_PHI3 A6
rf_finger index;

// Middle Finger Pins
# define MID_FSR_PIN A1
# define MID_SERV_PWM 3
# define MID_PHI1 A10
# define MID_PHI2 A5
# define MID_PHI3 A0
rf_finger middle;

// Thumb Pins
# define TH_FSR_PIN A3
# define TH_SERV_PWM 5
# define TH_PHI1 A7
# define TH_PHI2 A8
# define TH_PHI3 A9
rf_finger thumb;

double py_msg=0;

// Timing vars
# define READ_MSG 40
# define UPDATE_MOTORS  100
# define READ_SENSORS 50
unsigned long current_ts;
unsigned long elapsed_t;
unsigned long motor_ts;
unsigned long sensor_ts;
unsigned long msg_ts;

// Init serial comms class
Serial_Comms serial_comms;


//================ Keep this for now until I have a 2nd look at the servo encoder calibration
float index_servo_enc2deg(float enc_value){
  //float deg = ((47/238)*enc_value) + 63.98;
  float deg = (0.20225*enc_value) - 136.921;
  return deg;
}
float index_servo_enc2microsec(float enc){//this
  float microsec = (-4*enc)+3424;
  return microsec;
}
//==============



void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(500);

  // Initialise each finger with correct pins and calibration offsets for sensors
  index.initialise("I", INDEX_PHI1, INDEX_PHI2, INDEX_PHI3, INDEX_FSR_PIN, INDEX_SERV_PWM, 0.1935, -134.5161, 176.74, 239.56, -2.42, 2727.1);
  middle.initialise("M", MID_PHI1, MID_PHI2, MID_PHI3, MID_FSR_PIN, MID_SERV_PWM, 0.3475, -134.4788, 178.04, 218.73, -2.42, 2727.4);
  thumb.initialise("T", TH_PHI1, TH_PHI2, TH_PHI3, TH_FSR_PIN, TH_SERV_PWM, -0.3, 65.4, 177.07, 218.73, 2.42, 237.92);
}


void loop() {

  current_ts = millis();

  // Read the sensors
  elapsed_t = current_ts - sensor_ts;
  if (elapsed_t >= READ_SENSORS){
    index.read_encoders();
    index.read_fsr();

    middle.read_encoders();
    middle.read_fsr();

    thumb.read_encoders();
    thumb.read_fsr();
  }

  // send and recv data
  elapsed_t = current_ts - msg_ts;
  if (elapsed_t >= READ_MSG){
    serial_comms.recvWithStartEndMarkers();
    py_msg = atof(serial_comms.receivedChars);
    serial_comms.replyToPython(index.phi1, index.phi2, index.phi3, index.fsr_force,
                               middle.phi1, middle.phi2, middle.phi3, middle.fsr_force,
                               thumb.phi1, thumb.phi2, thumb.phi3, thumb.fsr_force,
                               middle.phi1_enc);
  }
  
  // Activate/deactivate the motors
  elapsed_t = current_ts - motor_ts;
  if (elapsed_t >= UPDATE_MOTORS){
    if (py_msg == 1){
      if (index.servo.attached()){ // if servo is active, maintain position
        index.servo.writeMicroseconds(index.fix_pos);
        // this is where the variable ff code will end up for the arduino side
      }
      if (middle.servo.attached()){ // if servo is active, maintain position
        middle.servo.writeMicroseconds(middle.fix_pos);
        // this is where the variable ff code will end up for the arduino side
      }
      if (thumb.servo.attached()){ // if servo is active, maintain position
        thumb.servo.writeMicroseconds(thumb.fix_pos);
        // this is where the variable ff code will end up for the arduino side
      }
      else{
        index.block();
        middle.block();
        thumb.block();
      }     
    }
    else if (py_msg==0){
      index.unblock();
      middle.unblock();
      thumb.unblock();
    }
  }
}
