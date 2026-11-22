/*
  Hareket kontrolcüsü — bilgisayardan seri port ile gelen komutları uygular.

  Komutlar (her biri tek karakter):
    A = avuç           -> röle AÇIK
    Y = yumruk         -> röle KAPALI
    I = işaret parmağı -> servo 30°
    V = iki parmak     -> servo 90° (orta)
    B = başparmak      -> servo 150°
    N = hareket yok    -> sadece ekranı günceller

  Bağlantılar:
    Servo sinyal -> D9      Röle IN -> D7
    LCD: RS->D12, E->D11, D4->D5, D5->D4, D6->D3, D7->D2
*/

#include <Servo.h>
#include <LiquidCrystal.h>

const int SERVO_PIN = 9;
const int RELAY_PIN = 7;

Servo servo;
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

bool relayOn = false;
int servoAngle = 90;

void showStatus(const char* gesture) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(gesture);
  lcd.setCursor(0, 1);
  lcd.print("Role:");
  lcd.print(relayOn ? "ON " : "OFF");
  lcd.print(" Srv:");
  lcd.print(servoAngle);
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  servo.attach(SERVO_PIN);
  servo.write(servoAngle);
  lcd.begin(16, 2);
  showStatus("Hazir");
}

void loop() {
  if (!Serial.available()) return;
  char cmd = Serial.read();

  switch (cmd) {
    case 'A': relayOn = true;  showStatus("Avuc");        break;
    case 'Y': relayOn = false; showStatus("Yumruk");      break;
    case 'I': servoAngle = 30;  showStatus("Isaret");     break;
    case 'V': servoAngle = 90;  showStatus("Iki parmak"); break;
    case 'B': servoAngle = 150; showStatus("Basparmak");  break;
    case 'N': showStatus("-");                            break;
    default: return;  // '\n' gibi bilinmeyen karakterleri yok say
  }

  // Not: Bazı röle modülleri ters çalışır (LOW = açık). Öyleyse HIGH/LOW'u yer değiştir.
  digitalWrite(RELAY_PIN, relayOn ? HIGH : LOW);
  servo.write(servoAngle);
}
