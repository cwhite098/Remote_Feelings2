
#include <Servo.h>  // add servo library
#include "serial_comms.h" //add serial class
#include <PID_v1.h>

# define FSR_PIN A2
# define INDEX_SERV_PWM 5
# define INDEX_POTPIN A7

int fsr_reading;
float force_reading;

// Init the stuff for the index servo
Servo indexservo;
int index_phi3_potpin = A4;
int index_phi2_potpin = A6;
double index_set_pos;
float index_phi1;
double Output;
float index_phi3;
float index_phi2;

double py_msg=0;
float pos;

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


float read_encoders(char finger){
  // pass in a character that corresponds to the finger you want to update
  if(finger = 'I'){
    analogReference(EXTERNAL);
    //index_phi1 = index_servo_enc2deg(analogRead(INDEX_POTPIN));
    index_phi1 = analogRead(INDEX_POTPIN);
    //index_actual_pos = analogRead(index_servo_potpin);

    analogReference(DEFAULT);
    index_phi3 = index_phi3_enc2deg(analogRead(index_phi3_potpin));
    index_phi2 = index_phi2_enc2deg(analogRead(index_phi2_potpin));
  }
  else{
  }
}

//================ trouble
float index_servo_enc2deg(float enc_value){
  //float deg = ((47/238)*enc_value) + 63.98;
  float deg = (0.31*enc_value)-154.3;
  return deg;
}

float index_servo_enc2microsec(float enc){
  float microsec = (-2.3793*enc)+2706.7;
  return microsec;
}

int index_servo_deg2microsec(float deg){
  //float deg = ((47/238)*enc_value) + 63.98;
  int microsec = (-11.302*deg) + 1422.4;
  return int(microsec);
}
//==============


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

  pinMode(INDEX_POTPIN, INPUT);
  pinMode(index_phi3_potpin, INPUT);
  pinMode(index_phi2_potpin, INPUT);

  pinMode(FSR_PIN, INPUT);

  indexservo.attach(INDEX_SERV_PWM);
  indexservo.writeMicroseconds(2000);
  delay(5000);
  indexservo.detach();

}


void loop() {

  current_ts = millis();


  // Read the sensors
  elapsed_t = current_ts - sensor_ts;
  if (elapsed_t >= READ_SENSORS){
    
  }

  // send and recv data
  elapsed_t = current_ts - msg_ts;
  if (elapsed_t >= READ_MSG){
    serial_comms.recvWithStartEndMarkers();
    py_msg = atof(serial_comms.receivedChars);
    serial_comms.replyToPython(index_servo_enc2deg(index_phi1), index_phi2, index_phi3, force_reading, int(pos));
  }
  
  // Activate/deactivate the motors
  elapsed_t = current_ts - motor_ts;
  if (elapsed_t >= UPDATE_MOTORS){
    
    read_encoders('I');
    fsr_reading = analogRead(FSR_PIN);
    force_reading = fsr_2N(fsr_reading);
    pos = index_servo_enc2microsec(index_phi1);
    
    if (py_msg == 1){
      if (indexservo.attached()){
        
        indexservo.writeMicroseconds(int(pos));
        //serial_comms.replyToPython(index_servo_enc2deg(index_phi1), index_phi2, index_phi3, force_reading, 69);
      }
      else{
        indexservo.attach(INDEX_SERV_PWM);
        indexservo.writeMicroseconds(int(pos));
      }
      
    }
    else if (py_msg==0){
      indexservo.detach();
      indexservo.writeMicroseconds(int(pos));
    }
  }
  

}
