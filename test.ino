#define LED 8         // LED connected to pin 8
#define BUZZER 5      // Buzzer connected to pin 5
#define TEMP_SENSOR A0 // Temperature sensor on A0

bool alarmActive = false;   // Flag to track alarm state
bool tempMonitorActive = false; // Flag to track temperature monitoring
unsigned long lastTempCheck = 0;
unsigned long lastAlarmToggle = 0;
const unsigned long tempInterval = 1000; // Check temperature every 1 second
const unsigned long alarmInterval = 500; // Change buzzer tone every 500 ms
int buzzerFrequency = 1000; // Initial buzzer frequency

void setup() {
  Serial.begin(9600);
  pinMode(LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'A') {
      activateAlarm();
    } else if (command == 'a') {
      deactivateAlarm();
    } else if (command == 'T') {
      tempMonitorActive = true;
    } else if (command == 't') {
      tempMonitorActive = false;
    }
  }

  if (tempMonitorActive && millis() - lastTempCheck >= tempInterval) {
    lastTempCheck = millis();
    checkTemperature();
  }

  if (alarmActive && millis() - lastAlarmToggle >= alarmInterval) {
    lastAlarmToggle = millis();
    toggleAlarmSound();
  }
}

void checkTemperature() {
  int sensorValue = analogRead(TEMP_SENSOR);
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperature = voltage * 100.0;
  Serial.println(temperature, 2);
}

void activateAlarm() {
  alarmActive = true;
  digitalWrite(LED, HIGH);  // Turn on LED
  tone(BUZZER, buzzerFrequency); // Start buzzer with initial frequency
}

void deactivateAlarm() {
  alarmActive = false;
  digitalWrite(LED, LOW);   // Turn off LED
  noTone(BUZZER);           // Turn off buzzer
}

void toggleAlarmSound() {
  // Alternate between two frequencies for a siren-like effect
  if (buzzerFrequency == 1000) {
    buzzerFrequency = 1500;
  } else {
    buzzerFrequency = 1000;
  }
  tone(BUZZER, buzzerFrequency); // Update buzzer frequency
}

 
  
