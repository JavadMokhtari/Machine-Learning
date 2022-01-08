#include <Wire.h>
#include "Adafruit_TCS34725.h"
#include <LiquidCrystal.h> 

/* Connect SCL    to analog 5
   Connect SDA    to analog 4
   Connect VDD    to 3.3V DC
   Connect GROUND to common ground */

// Initialise with default values (int time = 2.4ms, gain = 1x)
Adafruit_TCS34725 tcs = Adafruit_TCS34725();

// initialize the library with the numbers of the interface pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup(void) {
  Serial.begin(9600);
  lcd.begin(16, 2);
  lcd.setCursor(4, 0);
  lcd.print("Wellcome!");

}

void loop(void) {
  uint16_t r, g, b, c;
  tcs.getRawData(&r, &g, &b, &c);

  Serial.print(r, DEC); Serial.print(",");
  Serial.print(g, DEC); Serial.print(",");
  Serial.print(b, DEC); Serial.print(",");
  Serial.print("\n");
  delay(500);

  if (Serial.available()) {
    delay(100);  //wait some time for the data to fully be read
    lcd.clear();
    while (Serial.available() > 0) {
      char c = Serial.read();
      lcd.write(c);
    }
  }
}
