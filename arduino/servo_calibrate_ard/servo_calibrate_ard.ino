#include <Servo.h>  // add servo library

Servo servo;

int potpin = A11;
int pos;
int enc_pos;

char buffer[8];
String message;


void setup() {
  // put your setup code here, to run once:
  servo.attach(11);

  Serial.begin(9600);

  pinMode(potpin, INPUT);
  analogReference(EXTERNAL);
}

void loop() {
  // put your main code here, to run repeatedly:


  if (Serial.available() > 0) {
    // read byte of received data:
    int rlen = Serial.readBytesUntil('\n', buffer, 8);

    // prints the received data on serial monitor
    Serial.print(" Received Serial Data is: ");

    for(int i = 0; i < rlen; i++){
      Serial.print(buffer[i]);
    }

    Serial.println();
  }

  delay(100);


  
  
}
