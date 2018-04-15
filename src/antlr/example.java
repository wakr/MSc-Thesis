public class Ohjelma {
    public static void main(String[] args) {
        SalasananArpoja arpoja = new SalasananArpoja(13);
        System.out.println("Salasana: " + arpoja.luoSalasana());
        System.out.println("Salasana: " + arpoja.luoSalasana());
        System.out.println("Salasana: " + arpoja.luoSalasana());
        System.out.println("Salasana: " + arpoja.luoSalasana());
    }
}

import java.util.Random;

public class SalasananArpoja {
    private Random r;
    private int pituus;
    private final String s = "abcdefghijklmnopqrstuvwxyz";

    public SalasananArpoja(int salasananPituus) {
        this.r = new Random();
        this.pituus = salasananPituus;
    }

    public String luoSalasana() {
        String ss = "";
        for (int i = 0; i < pituus; i++) {
            ss += s.charAt(r.nextInt(26));
        }
        return ss;
    }
}
