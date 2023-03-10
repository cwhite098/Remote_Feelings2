#include <Servo.h>

class rf_finger {
  
  public:
    char f_name;
    
    // Exoskeleton angles
    float phi1;
    float phi2;
    float phi3;

    float phi1_enc;

    // Encoder pins
    int phi1_pin;
    int phi2_pin;
    int phi3_pin;
    int fsr_pin;
    int serv_pwm_pin;

    // FSR reading
    int fsr_reading;
    float fsr_force; 

    // Encoder offsets
    float phi1_offset;
    float phi1_grad;
    float phi2_offset;
    float phi3_offset;
    float enc_offset;
    float enc_grad;

    // Servo variables
    Servo servo;
    float pos; // the current position of the servo (microsec)
    int fix_pos; // the value to fix servo at when blocking
    
    // Constructor
    rf_finger(){
    }

    void initialise(char finger_name, int phi1_in, int phi2_in, int phi3_in, int fsr_in, int serv_pwm_in, 
                float phi1_m, float phi1_c, float phi2_c, float phi3_c, float enc2m_m, float enc2m_c){

      f_name = finger_name;
      
      // Save the values for the pins
      phi1_pin = phi1_in;
      phi2_pin = phi2_in;
      phi3_pin = phi3_in;
      fsr_pin = fsr_in;
      serv_pwm_pin = serv_pwm_in;
      
      // Call this in setup to get the right pin assignments
      pinMode(phi1_pin, INPUT);
      pinMode(phi2_pin, INPUT);
      pinMode(phi3_pin, INPUT);
      pinMode(fsr_pin, INPUT);

      phi1_offset = phi1_c;
      phi1_grad = phi1_m;
      phi2_offset = phi2_c;
      phi3_offset = phi3_c;

      enc_offset = enc2m_c;
      enc_grad = enc2m_m;

      servo.attach(serv_pwm_pin);
      delay(1000);
      servo.detach();
    }

    void read_encoders(){
      // Function to read the encoders for the finger
      analogReference(EXTERNAL);
      phi1 = servo_enc2deg(analogRead(phi1_pin));
      delay(10); // short delay after read to allow ref voltage change
      pos = servo_enc2microsec(analogRead(phi1_pin)); // save current servo position - brokken
      phi1_enc = analogRead(phi1_pin);
      
      analogReference(DEFAULT);
      phi3 = encoder_enc2deg(analogRead(phi3_pin), phi3_offset);
      delay(10); // short delay after read to allow ref voltage change
      phi2 = encoder_enc2deg(analogRead(phi2_pin), phi2_offset);
    }

    void read_fsr(){
      fsr_reading = analogRead(fsr_pin);
      fsr_force = fsr_2N(fsr_reading);
      //fsr_force=fsr_reading;
    }

    float encoder_enc2deg(int enc, float offset){
      // Convert the encoder readings into degrees (phi2 and phi3)
      float deg = ((-enc*333.3)/(1024.0)) + offset;
      return deg;
    }

    float servo_enc2deg(int enc_value){
      // Convert the servo encoder to degrees
      float deg = (phi1_grad*enc_value) + phi1_offset;
      return deg;
    }

    float servo_enc2microsec(int enc){
      // Convert servo encoder to microsecs to write to servo
      float microsec = (enc_grad*enc) + enc_offset;
      return microsec;
    }

    float fsr_2N(int fsr_reading){
      // rouhly convert fsr force to Newtons
      float force = 0.6867 * exp(0.0023*fsr_reading);
      return force;
    }

    void block(){
      // activate servo and save the position
      fix_pos = pos;
      servo.attach(serv_pwm_pin);
      servo.writeMicroseconds(fix_pos);
    }
    void unblock(){
      servo.detach();
    }

};
