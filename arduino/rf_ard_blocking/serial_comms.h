# define numChars 64

class Serial_Comms {

  public:

    // Init class vars
    //const byte numChars = 64;
    char receivedChars[numChars];
    boolean newData = false;

    // Constructor
    Serial_Comms(){
      
    }

    void recvWithStartEndMarkers() {
        static boolean recvInProgress = false;
        static byte ndx = 0;
        char startMarker = '<';
        char endMarker = '>';
        char rc;
    
        while (Serial.available() > 0 && newData == false) {
            rc = Serial.read();
    
            if (recvInProgress == true) {
                if (rc != endMarker) {
                    receivedChars[ndx] = rc;
                    ndx++;
                    if (ndx >= numChars) {
                        ndx = numChars - 1;
                    }
                }
                else {
                    receivedChars[ndx] = '\0'; // terminate the string
                    recvInProgress = false;
                    ndx = 0;
                    newData = true;
                }
            }
    
            else if (rc == startMarker) {
                recvInProgress = true;
            }
        }
    }
    
    //===============
    
    void replyToPython(float index_phi1, float index_phi2, float index_phi3, float index_F_f, 
                       float mid_phi1, float mid_phi2, float mid_phi3, float mid_F_f,
                       float th_phi1, float th_phi2, float th_phi3, float th_F_f,
                       float misc) {
        if (newData == true) {
            Serial.print("<");
            Serial.print(index_phi1);
            Serial.print(",");
            Serial.print(index_phi2);
            Serial.print(",");
            Serial.print(index_phi3);
            Serial.print(",");
            Serial.print(index_F_f);
            Serial.print(",");
            Serial.print(mid_phi1);
            Serial.print(",");
            Serial.print(mid_phi2);
            Serial.print(",");
            Serial.print(mid_phi3);
            Serial.print(",");
            Serial.print(mid_F_f);
            Serial.print(",");
            Serial.print(th_phi1);
            Serial.print(",");
            Serial.print(th_phi2);
            Serial.print(",");
            Serial.print(th_phi3);
            Serial.print(",");
            Serial.print(th_F_f);
            Serial.print(",");
            Serial.print(misc);
            Serial.print('>');
            newData = false;
        }
    }

  
};
