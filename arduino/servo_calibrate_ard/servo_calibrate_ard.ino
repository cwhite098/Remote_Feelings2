#include <Servo.h>  // add servo library
#include "serial_comms.h" //add serial class

Servo servo;
Serial_Comms serial_comms;
int pos=180;
int enc_pos;
char incomingByte;

# define PWM_PIN 11
# define POT_PIN A11

void setup() {
  // put your setup code here, to run once:
  servo.attach(PWM_PIN);

  Serial.begin(9600);

  pinMode(POT_PIN, INPUT);
  analogReference(EXTERNAL);
  servo.write(pos);
}

void loop() {
  // put your main code here, to run repeatedly:

    serial_comms.recvWithStartEndMarkers();

    servo.write(pos);
    analogReference(EXTERNAL);
    enc_pos = analogRead(POT_PIN);
    delay(10);

    
    serial_comms.replyToPython(pos, enc_pos);

    if (serial_comms.receivedChars[0] == 'c'){

      delay(10);
      pos = pos-10;
      incomingByte = 0;
      serial_comms.receivedChars[0] = 'x';
    }
  }
