
#include <Servo.h>  // add servo library
#include "serial_comms.h" //add serial class
#include <PID_v1.h>

# define FSR_PIN A2
# define INDEX_SERV_PWM 11

int fsr_reading;
float force_reading;

// Init the stuff for the index servo
Servo indexservo;
int index_servo_potpin = A11;
int index_phi3_potpin = A4;
int index_phi2_potpin = A6;
double index_set_pos;
float index_phi1;
double Output;
float index_phi3;
float index_phi2;

double py_msg=0;
int pos=0;

# define READ_MSG 10
# define UPDATE_MOTORS 10
# define READ_SENSORS 20

unsigned long current_ts;
unsigned long elapsed_t;
unsigned long motor_ts;
unsigned long sensor_ts;
unsigned long msg_ts;

// Init serial comms class
Serial_Comms serial_comms;


float read_encoders(char finger){
  // pass in a character that corresponds to the finger you want to update
  if(finger = 'I'){
    analogReference(EXTERNAL);
    index_phi1 = index_servo_enc2deg(analogRead(index_servo_potpin));
    //index_actual_pos = analogRead(index_servo_potpin);

    analogReference(DEFAULT);
    index_phi3 = index_phi3_enc2deg(analogRead(index_phi3_potpin));
    index_phi2 = index_phi2_enc2deg(analogRead(index_phi2_potpin));
  }
  else{
  }
}

float index_servo_enc2deg(float enc_value){
  //float deg = ((47/238)*enc_value) + 63.98;
  float deg = (enc_value/5.0623)-147.685;
  return deg;
}

int index_servo_deg2microsec(float deg){
  //float deg = ((47/238)*enc_value) + 63.98;
  int microsec = (12.115*(deg+45)) + 1326.6;
  return int(microsec);
}

float index_phi3_enc2deg(float enc_value){
  //float deg = ((47/238)*enc_value) + 63.98;
  float deg = ((-enc_value*360)/(1024)) + 193.359375;
  return deg;
}

float index_phi2_enc2deg(float enc_value){
  //float deg = ((47/238)*enc_value) + 63.98;
  float deg = ((-enc_value*360)/(1024)) + 175.78125;
  return deg;
}

float fsr_2N(int fsr_reading){
  float force = 0.6867 * exp(0.0023*fsr_reading);
  return force;
}


void setup() {
  // put your setup code here, to run once:
  //indexservo.attach(11);

  Serial.begin(115200);
  delay(500);

  pinMode(index_servo_potpin, INPUT);
  pinMode(index_phi3_potpin, INPUT);
  pinMode(index_phi2_potpin, INPUT);

  pinMode(FSR_PIN, INPUT);
}


void loop() {

  current_ts = millis();

  // Read the sensors
  elapsed_t = current_ts - sensor_ts;
  if (elapsed_t >= READ_SENSORS){
    read_encoders('I');
    fsr_reading = analogRead(FSR_PIN);
    force_reading = fsr_2N(fsr_reading);

    pos = index_servo_deg2microsec(index_phi1);
    
  }

  // send and recv data
  elapsed_t = current_ts - msg_ts;
  if (elapsed_t >= READ_MSG){
    serial_comms.recvWithStartEndMarkers();
    py_msg = atof(serial_comms.receivedChars);
    serial_comms.replyToPython(index_phi1, index_phi2, index_phi3, force_reading, pos);
  }
  
  // Activate/deactivate the motors
  elapsed_t = current_ts - motor_ts;
  if (elapsed_t >= UPDATE_MOTORS){
    if (py_msg == 1){
      
      indexservo.attach(INDEX_SERV_PWM);
      indexservo.write(int(index_phi1));
      
    }
    else if (py_msg==0){
      indexservo.detach();
    }
  }
  

}
