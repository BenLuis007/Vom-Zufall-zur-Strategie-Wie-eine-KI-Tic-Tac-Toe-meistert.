import random
from Neuronales_Netz_fuer_TicTacToe import Neuronales_Netz, Neuron
from Activation_functions import*

aktivierungsfunktion_hidden = sigmoid
aktivierungsfunktion_outp = ReLU
ableitung_hidden = ableitung_sigmoid
ableitung_outp = ableitung_ReLU


oben_links = [[0,1,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,1], [0,0,0,1,0,0,1,0,0]]
oben_mitte = [[1,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,1,0]]
oben_rechts = [[1,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,1,0,0], [0,0,0,0,0,1,0,0,1]]

mitte_links = [[1,0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,1,0]]
mitte_mitte = [[0,1,0,0,0,0,0,1,0], [0,0,0,1,0,1,0,0,0], [1,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,1,0,0]]
mitte_rechts = [[0,0,1,0,0,0,0,0,1], [0,0,0,1,1,0,0,0,0]]

unten_links = [[0,0,0,0,0,0,0,1,1], [0,0,1,0,1,0,0,0,0], [1,0,0,1,0,0,0,0,0]]
unten_mitte = [[0,0,0,0,0,0,1,0,1], [0,1,0,0,1,0,0,0,0]]
unten_rechts = [[0,0,0,0,0,0,1,1,0], [1,0,0,0,1,0,0,0,0], [0,0,1,0,0,1,0,0,0]]

nur_noch_ein_zug = [oben_links, oben_mitte, oben_rechts, mitte_links, mitte_mitte, mitte_rechts, unten_links, unten_mitte, unten_rechts]

wie_viel_gewonnen = [0,0,0,0,0,0,0,0,0]
wie_viel_verloren = [0,0,0,0,0,0,0,0,0]
epsilon = 0

def spielen(wie_viel, mit_print):
    global wie_viel_verloren, wie_viel_gewonnen
    for i in range(wie_viel):
        spielstand, wo_spielen = trainingsspielstand()
        outputs = n_Netz_1.output_berechnen(spielstand, aktivierungsfunktion_hidden, aktivierungsfunktion_outp)
        
        if ε_greedy_exploration(outputs[0]) == outputs[0][wo_spielen]:
            bewertung = 1
            if mit_print == True:
                print(bewertung,"ja", wo_spielen)
            wie_viel_gewonnen[wo_spielen] += 1
        else:
            bewertung = -1
            if mit_print == True:
                print(bewertung,"noooooooo", wo_spielen)
            wie_viel_verloren[wo_spielen] += 1
        n_Netz_1.trainieren(bewertung, [outputs], [spielstand], ableitung_hidden, ableitung_outp)

def trainingsspielstand():
    liste = random.choice(nur_noch_ein_zug)
    spielstand = list(random.choice(liste))
    welches_ist_das_richtige = nur_noch_ein_zug.index(liste)
    spielstand[nur_noch_ein_zug.index(liste)] = 99
    for wiederholungen in range(2):
        mögliche_felder = []
        for feld in range(len(spielstand)):
            if spielstand[feld] == 0:
                mögliche_felder.append(feld)
        ausgewählter_zug = random.choice(mögliche_felder)
        spielstand[ausgewählter_zug] = -1
    spielstand[nur_noch_ein_zug.index(liste)] = 0
    return spielstand, welches_ist_das_richtige
        
def ε_greedy_exploration(output_vom_NN):
    r = random.random()
    global epsilon
    epsilon += 0.000001 # chance das es den besten zug spielt wird immer grösser
    if r > (0.5 - epsilon) : # zu 80% spielt es die beste möglichkeit
        return max(output_vom_NN)
    if r > (0.3 - epsilon): #mit einer wahrscheinlichkeit von 12% spielt es den 2.besten zug
        return sorted(output_vom_NN)[-2]
    else: #mit einer wahrscheinlichkeit von 8% spielt es den 3.besten zug
        return sorted(output_vom_NN)[-3]

def trainingszyklen(wie_viele_episoden, wie_viele_spiele): # Mit diesen Episoden sieht man besser wie sich das Neuronale Netz verbessert.
    global wie_viel_verloren, wie_viel_gewonnen
    for episoden in range(wie_viele_episoden):
        wie_viel_gewonnen = [0,0,0,0,0,0,0,0,0]
        wie_viel_verloren = [0,0,0,0,0,0,0,0,0]
        spielen(wie_viele_spiele, False)
        print(wie_viel_verloren, "von", wie_viele_spiele, "spielen", sum(wie_viel_verloren), "mal verloren")
        print(wie_viel_gewonnen, "von", wie_viele_spiele, "spielen", sum(wie_viel_gewonnen), "mal gewonnen")
        
n_Netz_1 = Neuronales_Netz(9,9,9)
n_Netz_1.neuronales_Netz_erstellen()

#lässt das neuronale Netz trainieren
trainingszyklen(20, 50000)
