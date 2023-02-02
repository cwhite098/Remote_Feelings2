#include <Servo.h>  // add servo library
#include "serial_comms.h" //add serial class
#include <PID_v1.h>

# define FSR_PIN A2
int fsr_reading;
float force_reading;

// Init the stuff for the index servo
Servo indexservo;
int index_servo_potpin = A11;
int index_phi3_potpin = A4;
int index_phi2_potpin = A6;
double index_set_pos;
double index_actual_pos;
double Output;
float index_phi3;
float index_phi2;

double phi_d;

float start_time = millis();
float new_time=0;

// Init serial comms class
Serial_Comms serial_comms;
PID myPID(&index_actual_pos, &Output, &phi_d, 0.05,0,0, DIRECT);


float read_encoders(char finger){
  // pass in a character that corresponds to the finger you want to update
  if(finger = 'I'){
    analogReference(EXTERNAL);
    index_actual_pos = index_servo_enc2deg(analogRead(index_servo_potpin));
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

  index_set_pos = -45;
  

  indexservo.attach(11);
  indexservo.writeMicroseconds(index_servo_deg2microsec(-45));

  myPID.SetMode(AUTOMATIC);

}

void loop() {
  // put your main code here, to run repeatedly:

  
  float elapsed;
  // Send info for 10 seconds before using PID
  while (elapsed < 20000){
    
    elapsed = new_time - start_time;
    read_encoders('I');
    fsr_reading = analogRead(FSR_PIN);
    force_reading = fsr_2N(fsr_reading);
    
    serial_comms.recvWithStartEndMarkers();
    
    //phi_d = atof(serial_comms.receivedChars);
    serial_comms.replyToPython(index_actual_pos, index_phi2, index_phi3, force_reading, index_set_pos);
    
    new_time = millis();
  }
  
  read_encoders('I');
  fsr_reading = analogRead(FSR_PIN);
  force_reading = fsr_2N(fsr_reading);
  
  serial_comms.recvWithStartEndMarkers();
  
  phi_d = atof(serial_comms.receivedChars);
  serial_comms.replyToPython(index_actual_pos, index_phi2, index_phi3, force_reading, index_set_pos);

  myPID.Compute(); // compute the new servo position using PID
  index_set_pos = Output;
  //indexservo.writeMicroseconds(index_servo_deg2microsec(index_set_pos));

  // use phi_d from pyth in PID control loop

}
