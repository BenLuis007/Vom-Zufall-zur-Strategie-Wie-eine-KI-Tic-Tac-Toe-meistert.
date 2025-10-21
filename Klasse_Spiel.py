import random
from Activation_functions import*

aktivierungsfunktion_hidden = sigmoid
aktivierungsfunktion_outp = ReLU
ableitung_hidden = ableitung_sigmoid
ableitung_outp = ableitung_ReLU

class Spiel:
    def __init__(self, spieler_1, spieler_2):
        self.spieler_1 = spieler_1   #hat spielsteine +1
        self.spieler_2 = spieler_2  #hat spielsteine -1
        self.spielstand = [0,0,0,0,0,0,0,0,0]
        self.gespielte_züge_spieler_1 = []
        self.gespielte_züge_spieler_2 = []
        self.spielzüge = []
        self.wie_viel_verloren = 0 #zählt die verlorenen Spiele in einer Trainingsphase von Spieler_1
        self.wie_viel_gewonnen = 0
        self.wie_viel_unentschieden = 0
        self.trainingsspiel = True # self.trainingsspiel ist True wenn die KI trainiert wird, nach dem Training wenn sie gegen einen Mensch spielt ist es False.    
        
    def __repr__(self):    
        return f"{self.spielstand[0]} ,  {self.spielstand[1]},  {self.spielstand[2]}/n {self.spielstand[3]}, {self.spielstand[4]}, {self.spielstand[5]} /n {self.spielstand[6]}, {self.spielstand[7]}, {self.spielstand[8]}"
        
    def spiel_KI_KI(self, spielstand, anzahl_vergangene_züge, aktueller_spieler): # wenn beide einmal gestpielt haben ist anzahl züge 2
        self.trainingsspiel = True
        self.spielstand = spielstand
        self.aktueller_spieler = aktueller_spieler
        output_NN_spieler_1 = []
        output_NN_spieler_2 = []
        gegebene_inputs = [0, [], []] #[1] sind die inputs des 1. spielers, [2] sind die inputs von spieler 2
        
        if self.aktueller_spieler == self.spieler_1:
            self.spielerwert = 1
        else:
            self.spielerwert = -1
            
        for zug in range(9 - anzahl_vergangene_züge): 
            
            gegebene_inputs[self.spielerwert].append(self.spielstand)
            
            output_NN, bewertung = self.spielzug_KI(self.spielerwert, self.aktueller_spieler)
            
            #Der Spieler wird gewechselt, der nächste ist dran
            if self.aktueller_spieler == self.spieler_1: 
                self.aktueller_spieler = self.spieler_2
                self.spielerwert = -1
                output_NN_spieler_1.append(output_NN)
            else:
                self.aktueller_spieler = self.spieler_1
                self.spielerwert = 1
                output_NN_spieler_2.append(output_NN)
                
            if bewertung != 0:
                return [bewertung, output_NN_spieler_1, output_NN_spieler_2, gegebene_inputs]
            
        #wenn alle 9 Züge gespielt sind ohne dass jemand gewann, dann ist unentschieden    
        self.wie_viel_unentschieden += 1  
        return [bewertung, output_NN_spieler_1, output_NN_spieler_2, gegebene_inputs]

    def spiel_gegen_zufall(self, spielstand, spieler, spielerwert):
        self.trainingsspiel = True
        mögliche_züge = []
        output_NN = []
        gegebene_inputs = []
        self.spielstand = spielstand
        self.spielerwert = spielerwert
        
        for zug in range(9-5):
            gegebene_inputs.append(self.spielstand)
            output_NN_einzeln, bewertung = self.spielzug_KI(1, spieler)
            output_NN.append(output_NN_einzeln)
            
            if bewertung != 0:
                return [bewertung, output_NN, gegebene_inputs]
            
            for feld in range(len(self.spielstand)):
                if self.spielstand[feld] == 0:
                    mögliche_züge.append(feld)
            spielzug_random = random.choice(mögliche_züge)
            self.spielstand[spielzug_random] = self.spielerwert *-1
            bewertung = self.hat_gewonnen(self.spielerwert*-1)
            
            if bewertung != 0:
                return [bewertung, output_NN, gegebene_inputs]
            
        # der 9te Spielzug (von der KI)
        gegebene_inputs.append(self.spielstand)
        output_NN_einzeln, bewertung = self.spielzug_KI(1, spieler)
        output_NN.append(output_NN_einzeln)
        if bewertung != 0:
                return [bewertung, output_NN, gegebene_inputs]
        else:
            self.wie_viel_unentschieden += 1
            return [bewertung, output_NN, gegebene_inputs]
        
    def spiel_gegen_mensch(self, spieler, darf_KI_anfangen): #darf_KI_anfangen = True = ja
        #Mensch hat spielsteine -1, KI hat 1
        self.trainingsspiel = False #das heisst, kein Epsilon Greedy wird gebraucht
        self.spielstand = [0,0,0,0,0,0,0,0,0]
        print("Du tritst gegen eine KI an, schaffst du es sie zu besiegen?")
        
        if darf_KI_anfangen == False:
            self.mensch_spielen_lassen()
        
        for i in range (4):
            outp_gewonnen = self.spielzug_KI(1, spieler)
            if outp_gewonnen[1] == -100:
                print("Die KI hat leider einen Illegalen Zug gespielt, das heisst an einen Ort wo schon ein Spielstein war. Der Spielstand vor dem Illegalen Zug war wie folgt: ", self.spielstand)
                return
            elif outp_gewonnen[1] == 1:
                print("Die KI hat leider gewonnen, vielleicht klappts beim nächsten mal.", self.spielstand)
                
            self.mensch_spielen_lassen()
            if self.hat_gewonnen(-1) == -1:
                print("Bravo, du hast gewonnen", self.spielstand)
            
        if darf_KI_anfangen ==True:
            outp_gewonnen = spielzug_KI(1, spieler)
            if outp_gewonnen[1] == -100:
                print("Die KI hat leider einen Illegalen Zug gespielt, das heisst an einen Ort wo schon ein Spielstein war. Der Spielstand vor dem Illegalen Zug war wie folgt:", self. spielstand)
                return
            elif outp_gewonnen[1] == 1:
                print("Die KI hat leider gewonnen, vielleicht klappts beim nächsten mal.", self.spielstand)
                
        print("Unentschieden, niemand hat gewonnen", self.spielstand)
        return
        
    def spielzug_KI(self, spielerwert, spieler):
        output_NN = spieler.output_berechnen(self.spielstand, aktivierungsfunktion_hidden, aktivierungsfunktion_outp)
        bewertung = self.spielzug_auswerten(output_NN[0], spielerwert)
        return [output_NN, bewertung]
    
    def mensch_spielen_lassen(self):
        mögliche_züge = []
        antwort_spielzug = ""
        antwort_spielzug_int = 1000
        
        for feld in range(len(self.spielstand)):
                if self.spielstand[feld] == 0:
                    mögliche_züge.append(feld)
        
        while antwort_spielzug_int not in mögliche_züge:
            antwort_spielzug = ""
            while antwort_spielzug not in ("0","1","2","3","4","5","6","7","8"):
                print(self.spielstand[0],self.spielstand[1],self.spielstand[2])
                print(self.spielstand[3],self.spielstand[4],self.spielstand[5])
                print(self.spielstand[6],self.spielstand[7],self.spielstand[8])
                antwort_spielzug = input("wo willst du spielen?")
                antwort_spielzug_int = int(antwort_spielzug)
        self.spielstand[antwort_spielzug_int] = -1
        return int(antwort_spielzug)
                
    def spielzug_auswerten(self, output_vom_NN, spielerwert):
        #Wenn es ein Spiel zum training der KI ist, dann wird manchmal mit ε_greedy_exploration einen zufallszug gewählt
        if self.trainingsspiel == True: 
            ausgewählter_wert = self.ε_greedy_exploration(output_vom_NN)
            spielzug = output_vom_NN.index(ausgewählter_wert)    
        else:
            grösster_wert = max(output_vom_NN)
            spielzug = output_vom_NN.index(grösster_wert)
        self.spielzüge.append(spielzug)
    
        if self.spielstand[spielzug] != 0: #Wenn die KI einen Illegalen Zug spielen will geht es nocht, bricht es das spiel ab
            return -100 * spielerwert # Es wird mal den Spielerwert gerechnet, dass man auch weis welcher spieler falsch gespielt hat (nur für die Funktion spiel_KI_KI wichtig)
        else:
            self.spielstand[spielzug] = spielerwert
            return self.hat_gewonnen(spielerwert)
        
    def hat_gewonnen(self, spielerwert):
        
        for i in range(3): # zuerst die vertikale Achse dann nach dem or die horizontale achse
            if self.spielstand[i] + self.spielstand[i+3] + self.spielstand[i+6] == 3 * spielerwert or self.spielstand[i*3] + self.spielstand[i*3+1] + self.spielstand[i*3+2] == 3 * spielerwert:
                        
                if self.trainingsspiel == True:
                    if spielerwert == 1:
                        self.wie_viel_gewonnen += 1
                    else:
                        self.wie_viel_verloren += 1
                        
                return spielerwert #sagt wer gewonnen hat... 1/-1
            
        for i in range(2): #diagonale achsen
            if self.spielstand[i*2] + self.spielstand[4] + self.spielstand[8-i*2] == 3 * spielerwert:
                        
                if self.trainingsspiel == True:
                    if spielerwert == 1:
                        self.wie_viel_gewonnen += 1
                    else:
                        self.wie_viel_verloren += 1
                        
                return spielerwert
        return 0
    
    def ε_greedy_exploration(self, output_vom_NN):
        r = random.random()
        self.spieler_1.epsilon += 0.00001 # chance das es den besten zug spielt wird immer grösser
        if r > (0.5 - self.spieler_1.epsilon) : # zu 80% spielt es die beste möglichkeit
            return max(output_vom_NN)
        if r > (0.3 - self.spieler_1.epsilon): #mit einer wahrscheinlichkeit von 12% spielt es den 2.besten zug
            return sorted(output_vom_NN)[-2]
        else: #mit einer wahrscheinlichkeit von 8% spielt es den 3.besten zug
            return sorted(output_vom_NN)[-3]