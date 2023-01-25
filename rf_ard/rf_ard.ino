#include <Servo.h>  // add servo library

// Init the message receiving stuff
char buffer[8];
String message;

// Init the stuff for the index servo
Servo indexservo;
int index_servo_potpin = A11;
int index_phi3_potpin = A4;
int index_phi2_potpin = A6;
int index_set_pos;
float index_actual_pos;
float index_phi3;
float index_phi2;


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
  float microsec = (12.115*deg)/3090.35;
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




void setup() {
  // put your setup code here, to run once:
  //indexservo.attach(11);

  Serial.begin(9600);

  pinMode(index_servo_potpin, INPUT);
  pinMode(index_phi3_potpin, INPUT);
  pinMode(index_phi2_potpin, INPUT);

  indexservo.attach(11);
  indexservo.writeMicroseconds(1500);
}

void loop() {
  // put your main code here, to run repeatedly:

  read_encoders('I');

  
  if (Serial.available() > 0) {
    // read byte of received data:
    int rlen = Serial.readBytesUntil('\n', buffer, 8);

    // prints the received data on serial monitor
    Serial.print(index_actual_pos);
    Serial.print(',');
    Serial.print(index_phi2);
    Serial.print(',');
    Serial.print(index_phi3);
    Serial.println();
    delay(10);
    
    //for(int i = 0; i < rlen; i++){
      //Serial.print(buffer[i]);
    //}
  }

}
