import json, os
from Neuronales_Netz_fuer_TicTacToe import Neuronales_Netz, Neuron
from Activation_functions import*
from Klasse_Spiel import Spiel

aktivierungsfunktion_hidden = sigmoid
aktivierungsfunktion_outp = ReLU
ableitung_hidden = ableitung_sigmoid
ableitung_outp = ableitung_ReLU

      
def spiel_KI_gegen_KI(KI1, KI2, anzahl_spiele): #KI spielt gegen eine andere KI, beide werden dabei trainiert
    s = Spiel(KI1, KI2)
    for i in range(anzahl_spiele):
        spielstand = [0,0,0,0,0,0,0,0,0]
        bewertung = s.spiel_KI_KI(spielstand, 0, KI1)
        KI1.trainieren(bewertung[0], bewertung[1], bewertung[3][1], ableitung_hidden, ableitung_outp)
        
        KI2.trainieren((bewertung[0] * -1), bewertung[2], bewertung[3][2], ableitung_hidden, ableitung_outp) #bewertung * -1 ist weil die bewertung von spieler 2 immer umgekehrt ist (-1 = gewonnen...)
        # bewertung *-1 macht das dann die -100 bewertng zu bewertung 100 wird...
        if i % 10 == 0: #nicht alle spiele werden angezeigt, aber dass man doch noch eine Ahnung hat, wie gespielt wurde wird der Spielstand manchmal ausgegeben.
            print(spielstand)
    print("von", anzahl_spiele, "spielen", s.wie_viel_gewonnen, "gewonnen und", s.wie_viel_verloren, "verloren und", s.wie_viel_unentschieden, "unentschieden")


def spiel_KI_gegen_zufall(KI1, anzahl_spiele, spielerwert): #KI spielt gegen den Zufall und wird nach jedem Spiel trainiert
    s = Spiel(KI1, KI1)
    for i in range(anzahl_spiele):
        spielstand = [0,0,0,0,0,0,0,0,0]
        bewertung = s.spiel_gegen_zufall(spielstand, KI1, spielerwert)
        KI1.trainieren(bewertung[0], bewertung[1], bewertung[2], ableitung_hidden, ableitung_outp)  
    print("von", anzahl_spiele, "spielen", s.wie_viel_gewonnen, "gewonnen und", s.wie_viel_verloren, "verloren und", s.wie_viel_unentschieden, "unentschieden")
    
    
def spiel_KI_gegen_Mensch(KI1, anzahl_spiele): #KI spielt ein Spiel gegen de Mensch, ohne dabei trainiert zu weden
    s = Spiel(KI1,KI1)
    for i in range(anzahl_spiele):
        s.spiel_gegen_mensch(KI1, False)
        


def datei_speichern(dateinamen):
    desktop_pfad = os.path.join(os.path.expanduser("~"), "Desktop") # speicherort Pfad automatisch herausfinden
    datei_pfad = os.path.join(desktop_pfad, dateinamen)

    #  Neuronen in dicts umwandeln
    neuronen_dicts_1 = [{"weights": n.weights_n, "bias": n.weight_bias} for n in n_Netz_1.hidden_layer_1]
    neuronen_dicts_2 = [{"weights": n.weights_n, "bias": n.weight_bias} for n in n_Netz_1.hidden_layer_2]
    neuronen_dicts_3 = [{"weights": n.weights_n, "bias": n.weight_bias} for n in n_Netz_1.output_layer]

    #in JSON speichern
    with open(datei_pfad, "w") as datei:
        json.dump([neuronen_dicts_1,neuronen_dicts_2, neuronen_dicts_3], datei)
def datei_öffnen(dateinamen):
    desktop_pfad = os.path.join(os.path.expanduser("~"), "Desktop") # speicherort Pfad automatisch herausfinden
    datei_pfad = os.path.join(desktop_pfad, dateinamen)
    
    with open(datei_pfad, "r") as f: #datei öffnen
        geladene_dicts = json.load(f)

    # wieder Neuron objekte erzeugen
    n_Netz_1.hidden_layer_1 = [Neuron(d["bias"], d["weights"]) for d in geladene_dicts[0]]
    n_Netz_1.hidden_layer_2 = [Neuron(d["bias"], d["weights"]) for d in geladene_dicts[1]]
    n_Netz_1.output_layer = [Neuron(d["bias"], d["weights"]) for d in geladene_dicts[2]]
      
       

n_Netz_1 = Neuronales_Netz(9,9,13)
n_Netz_1.neuronales_Netz_erstellen()

n_Netz_2 = Neuronales_Netz(9,9,9)
n_Netz_2.neuronales_Netz_erstellen()


datei_öffnen("NN_Gewichte_backpropagation")

#2 neuronale Netze werden seperat gegen den Zufall trainiert.
for i in range(10):
    spiel_KI_gegen_zufall(n_Netz_1, 500, 1)
    
for i in range(10):
    spiel_KI_gegen_zufall(n_Netz_2, 500, -1)

#Die 2 neuronalen Netze werden nun trainiert indem sie gegeneinander Spielen.
for i in range(20):
    spiel_KI_gegen_KI(n_Netz_1,n_Netz_2, 100)

#Das erste neuronale Netz tritt nun gegen einen Mensch an.
spiel_KI_gegen_Mensch(n_Netz_1, 5)

#datei_speichern("NN_Gewichte_backpropagation")
    




'''legende:
0 = nimand besitzt das Feld
1 = KI gehört das Feld
-1 = gegner hat das Feld

Die Liste Spielstand:
Welche Zahl muss man eingeben um an einem besteimmten Ort zu spielen?

    [0,1,2
     3,4,5
     6,7,8]

'''
