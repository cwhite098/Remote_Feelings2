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
double index_phi1;
double Output;
float index_phi3;
float index_phi2;

double py_msg;

float start_time = millis();
float new_time=0;

// Init serial comms class
Serial_Comms serial_comms;
//PID myPID(&index_actual_pos, &Output, &phi_d, 0.05,0,0, DIRECT);


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

float index_servo_deg2microsec(float deg){
  //float deg = ((47/238)*enc_value) + 63.98;
  float microsec = (12.115*(deg+45)) + 1326.6;
  return microsec;
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

  pinMode(index_servo_potpin, INPUT);
  pinMode(index_phi3_potpin, INPUT);
  pinMode(index_phi2_potpin, INPUT);

  pinMode(FSR_PIN, INPUT);

  
}

void loop() {

  read_encoders('I');
  fsr_reading = analogRead(FSR_PIN);
  force_reading = fsr_2N(fsr_reading);
  
  serial_comms.recvWithStartEndMarkers();
  
  py_msg = atof(serial_comms.receivedChars);
  serial_comms.replyToPython(index_phi1, index_phi2, index_phi3, force_reading, index_phi1);

  if (py_msg == 1){
    indexservo.attach(INDEX_SERV_PWM);
    indexservo.writeMicroseconds(index_servo_deg2microsec(index_phi1));
  }
  else if (py_msg==0){
    indexservo.detach();
  }


  // use phi_d from pyth in PID control loop

}
