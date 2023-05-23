#include <Servo.h>  // add servo library
# define PWM_PIN 3
# define POT_PIN A10
Servo servo;


void setup() {
  // put your setup code here, to run once:
  servo.attach(PWM_PIN);

  Serial.begin(9600);

  pinMode(POT_PIN, INPUT);
  analogReference(EXTERNAL);
}

void loop() {
  servo.writeMicroseconds(1500);
  delay(500);
  Serial.println(1500);
  int enc_pos = analogRead(POT_PIN);
  Serial.println(enc_pos);
  delay(1000);
  
  
  servo.writeMicroseconds(2000);
  delay(500);
  Serial.println(2000);
  enc_pos = analogRead(POT_PIN);
  Serial.println(enc_pos);
  delay(1000);

  servo.writeMicroseconds(1200);
  delay(500);
  Serial.println(1200);
  enc_pos = analogRead(POT_PIN);
  Serial.println(enc_pos);
  delay(1000);

}
