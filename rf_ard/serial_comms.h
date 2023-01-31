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
    
    void replyToPython(float phi1, float phi2, float phi3, float F_f) {
        if (newData == true) {
            Serial.print("<");
            Serial.print(phi1);
            Serial.print(",");
            Serial.print(phi2);
            Serial.print(",");
            Serial.print(phi3);
            Serial.print(",");
            //Serial.print(F_f);
            Serial.print(receivedChars);
            Serial.print('>');
            newData = false;
        }
    }

  
};
