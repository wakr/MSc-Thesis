import easyIO.*;
// Oblig 2 [...]
//Jeg lasta inn programmskissen som ble gitt i oppgaveteksten og
//bygde videre fra den.
class Olje {
 In tast = new In();
 Out skjerm = new Out();
 String[][] eier = new String[10][15];
 int[][] oljeutvinnet = new int[10][15];
 void finnRadenMedHoystOljeutvinning() {
  //Går gjennom radene for å finne hvilken rad som har
  //høyest utvinning
  int rad = 0;
  int sum = 0;
  int temp = 0;
  for (int i = 0; i < oljeutvinnet.length; i++) {
   for (int k = 0; k < oljeutvinnet[i].length; k++) {
    temp += oljeutvinnet[i][k];
   }
   if (temp > sum) {
    sum = temp;
    rad = i;
   }
  }
  System.out.println("Rad med høyest oljeutvinning: rad " + rad + "(" + sum + " fat)");
 }
 void oppdaterOljeutvinning() {
  //Bruker samme for løkke som i forrige oppgave, og printer
  //ut teksten om utvunnet olje. Så spør jeg bruker om hvor
  //mange fat som ble utvunnet.
  for (int i = 0; i < eier.length; i++) {
   for (int k = 0; k < eier[i].length; k++) {
    if (eier[i][k] != null) {
     skjerm.out("Antall fat utvunnet i felt " + i +
      "-" + k + " siste 6 mnd. (tidligere total " + olkeutvinnet[i][k] + " fat): ");
     int fat = tast.inInt();
     oljeutvinnet[i][k] += fat;
    }
   }
  }
 }
}