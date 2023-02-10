#include <Servo.h>  // add servo library

Servo servo;

int potpin = A11;
int pos=2500;
int enc_pos;
char incomingByte;


void setup() {
  // put your setup code here, to run once:
  servo.attach(11);

  Serial.begin(9600);

  pinMode(potpin, INPUT);
  analogReference(EXTERNAL);
  servo.writeMicroseconds(pos);
}

void loop() {
  // put your main code here, to run repeatedly:

  if (Serial.available() > 0) {
    // read the incoming byte:
    incomingByte = Serial.read();

    // say what you got:
    Serial.println();

    if (incomingByte = 'c'){

      servo.writeMicroseconds(pos);
      Serial.print("Microseconds:");
      Serial.print(" ");
      Serial.print(pos);

      delay(1000);

      enc_pos = analogRead(potpin);
      Serial.print("Enc Value:");
      Serial.print(" ");
      Serial.print(enc_pos);

      pos = pos-100;
      incomingByte = 0;
    }
  }
}
