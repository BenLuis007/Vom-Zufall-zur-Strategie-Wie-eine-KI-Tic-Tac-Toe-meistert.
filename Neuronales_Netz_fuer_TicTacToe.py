import random
from Activation_functions import*

lr = 0.005

class Neuronales_Netz:
    def __init__(self, anzahl_input, anzahl_output, anzahl_connectors):
        self.anzahl_input = anzahl_input
        self.anzahl_output = anzahl_output
        self.anzahl_connectors = anzahl_connectors
        self.hidden_layer_1 = []#Liste mit Neuronen, die jeweils die gewichte vom "vorderen" layer beinhalten 
        self.hidden_layer_2 = []
        self.output_layer = [] #Liste mit Neuronen, die jeweils die gewichte vom "vorderen" layer beinhalten 
        self.epsilon = 0 #wird gebraucht um den zufalls faktor immer weiter zu senken
        
    def __repr__(self):
        return self.hidden_layer_1 + self.hidden_layer_2 + self.output_layer
            
    def layers_mit_neuronen_erstellen(self, anzahl_neuronen_im_layer_vorher, anzahl_neuronen): #erstellt Neuronen mit zufälligen Gewichten
        neuronen_layer = []
        for i in range(anzahl_neuronen):
            neuronen_gewichte = []
            for i in range(anzahl_neuronen_im_layer_vorher):
                neuronen_gewichte.append(random.random()- 0.5) 
            neuronen_layer.append(Neuron(0, neuronen_gewichte)) #verbessern
        return neuronen_layer
        
    def neuronales_Netz_erstellen(self): #erstellt layer die mit neuronen befüllt werden
        self.hidden_layer_1 = self.layers_mit_neuronen_erstellen(self.anzahl_input, self.anzahl_connectors)
        self.hidden_layer_2 = self.layers_mit_neuronen_erstellen(self.anzahl_connectors, self.anzahl_connectors)
        self.output_layer = self.layers_mit_neuronen_erstellen(self.anzahl_connectors, self.anzahl_output)
    
    def einzelner_output_berechnen(self, inputs, neuronen_layer, aktivierungsfunktion):
        self.inputs = inputs
        self.neuronen_layer = neuronen_layer
        output_einzeln = []
        output_zusammen = []
        
        for neuron in neuronen_layer:
            for i, weight in enumerate(neuron.weights_n):
                output_einzeln.append(weight * self.inputs[i])
            output_einzeln.append(neuron.weight_bias)
            output_neuron = aktivierungsfunktion(round(sum(output_einzeln), 7)) #round() damit bei sigmoid funktion nicht eine zahl sehr nahe an 0 kommt
            output_zusammen.append(output_neuron)
            output_einzeln = []
        return output_zusammen
    
    def output_berechnen(self, inputs, aktivierungsfunktion_hidden, aktivierungsfunktion_outp):
        
        outp_neuronen_hidden_layer_1 = self.einzelner_output_berechnen(inputs, self.hidden_layer_1, aktivierungsfunktion_hidden)
        outp_neuronen_hidden_layer_2 = self.einzelner_output_berechnen(outp_neuronen_hidden_layer_1, self.hidden_layer_2, aktivierungsfunktion_hidden)
        outp_neuronen_outp_layer = self.einzelner_output_berechnen(outp_neuronen_hidden_layer_2, self.output_layer, aktivierungsfunktion_outp)
        return [outp_neuronen_outp_layer, outp_neuronen_hidden_layer_1, outp_neuronen_hidden_layer_2]
        
    def trainieren(self, bewertung, ausgerechneter_outp_und_hidden_layer, gegebener_input, ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp):
        ausgerechneter_outp = []
        ausgerechneter_hidden_layer_1 = []
        ausgerechneter_hidden_layer_2 = []
        gespielte_züge = [] #auf welches feld wurde gespielt
        
        for züge in ausgerechneter_outp_und_hidden_layer:
            ausgerechneter_outp.append(züge[0])
            ausgerechneter_hidden_layer_1.append(züge[1])
            ausgerechneter_hidden_layer_2.append(züge[2])
            gespielte_züge.append(züge[0].index(max(züge[0])))
    
        if bewertung == 1: #Das NN hat gewonnen, alle gespielten züge werden verbessert
            for aktueller_zug, zug_feld in enumerate(gespielte_züge):
                optimaler_outp =[]
                for i in range(self.anzahl_output):
                    optimaler_outp.append(ausgerechneter_outp[aktueller_zug][i] * 0.9)
                optimaler_outp[zug_feld] = ausgerechneter_outp[aktueller_zug][zug_feld] * 1.25
                self.backpropagation(optimaler_outp, ausgerechneter_outp[aktueller_zug], ausgerechneter_hidden_layer_1[aktueller_zug], ausgerechneter_hidden_layer_2[aktueller_zug], gegebener_input[aktueller_zug], ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp)
            return
        
        if bewertung == -1: # Das NN hat verloren, alle gespielten züge werden schlechter gemacht, sodass sie nicht mehr gespielt werden
            for aktueller_zug, zug_feld in enumerate(gespielte_züge):
                optimaler_outp =[]
                for i in range(self.anzahl_output):
                    optimaler_outp.append(ausgerechneter_outp[aktueller_zug][i] * 1.1)
                optimaler_outp[zug_feld] =  0.75 * ausgerechneter_outp[aktueller_zug][zug_feld]
                self.backpropagation(optimaler_outp, ausgerechneter_outp[aktueller_zug], ausgerechneter_hidden_layer_1[aktueller_zug], ausgerechneter_hidden_layer_2[aktueller_zug], gegebener_input[aktueller_zug], ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp)
            return
        
        if bewertung == -100: # Das NN hat einen illegalen Zug gespielt, nur der für den Letzten Zug wird das Netz trainiert
            letzter_zug_feld = gespielte_züge[-1]
            aktueller_letzter_zug = gespielte_züge.index(letzter_zug_feld) #der wievielte zug war der letzte?
            optimaler_outp =[]
            for i in range(self.anzahl_output):
                optimaler_outp.append(ausgerechneter_outp[aktueller_letzter_zug][i] * 1.05) # alle nicht gespielten züge werden mit dem Faktor 1.01 besser gemacht
            optimaler_outp[letzter_zug_feld] =  0.9 * ausgerechneter_outp[aktueller_letzter_zug][letzter_zug_feld] # der letzte gespielte Zug, der illegal war wir schlechter gemacht
            self.backpropagation(optimaler_outp, ausgerechneter_outp[aktueller_letzter_zug], ausgerechneter_hidden_layer_1[aktueller_letzter_zug], ausgerechneter_hidden_layer_2[aktueller_letzter_zug], gegebener_input[aktueller_letzter_zug], ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp)
            return
        
        if bewertung == 100: # Der Gegner hat falsch gespielt
            # Das NN wird nicht angepasst da man nicht weis ob gut oder schlecht gespielt wurde
            return
            
        if bewertung == 0: # Unentschieden, alle gespielten züge werden leicht verbessert
            for aktueller_zug, zug_feld in enumerate(gespielte_züge):
                optimaler_outp =[]
                for i in range(self.anzahl_output):
                    optimaler_outp.append(ausgerechneter_outp[aktueller_zug][i] * 0.98)
                optimaler_outp[zug_feld] = ausgerechneter_outp[aktueller_zug][zug_feld] * 1.1
                self.backpropagation(optimaler_outp, ausgerechneter_outp[aktueller_zug], ausgerechneter_hidden_layer_1[aktueller_zug], ausgerechneter_hidden_layer_2[aktueller_zug], gegebener_input[aktueller_zug], ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp)
            return    
                    
    def backpropagation(self, optimaler_output, ausgerechneter_output, ausgerechnete_werte_hidden_layer_1, ausgerechnete_werte_hidden_layer_2, gegebener_input, ableitung_aktivierungsfunktion_hidden, ableitung_aktivierungsfunktion_outp):
        relativer_fehler_output = []
        relativer_fehler_hidden_1 = []
        relativer_fehler_hidden_2 = []
        abgeleiteter_fehler = 0
        
        #Für jedes Neuron in jeder Schicht hat es einen Fehler, der am anfang noch Null ist.
        for i in range(self.anzahl_connectors): 
            relativer_fehler_hidden_1.append(0)
            relativer_fehler_hidden_2.append(0)
            
        #output_layer verbessern/trainieren
        for feld, neuron in enumerate(self.output_layer):
            relativer_fehler_output.append(optimaler_output[feld] - ausgerechneter_output[feld])
            for i, weight in enumerate(neuron.weights_n):
                relativer_fehler_hidden_2[i] += relativer_fehler_output[feld] * weight/ sum(neuron.weights_n)
                abgeleiteter_fehler = relativer_fehler_output[feld] * ableitung_aktivierungsfunktion_outp(ausgerechneter_output[feld]) * ausgerechnete_werte_hidden_layer_2[i]

                neuron.weights_n[i] += lr * abgeleiteter_fehler # Gewicht anpassen
            neuron.weight_bias += relativer_fehler_output[feld] * lr  # Bias anpassen
            
        #hidden_layer_2 verbessern, trainieren
        for feld, neuron in enumerate(self.hidden_layer_2):
            for i, weight in enumerate(neuron.weights_n):
                relativer_fehler_hidden_1[i] += relativer_fehler_hidden_2[feld] * weight/ sum(neuron.weights_n)
                abgeleiteter_fehler = relativer_fehler_hidden_2[feld] * ableitung_aktivierungsfunktion_hidden(ausgerechnete_werte_hidden_layer_2[feld])  * ausgerechnete_werte_hidden_layer_1[feld]

                neuron.weights_n[i] += lr * abgeleiteter_fehler # Gewicht anpassen
            neuron.weight_bias += relativer_fehler_hidden_2[feld] * lr # Bias anpassen
            
        #hidden_layer_1 verbessern, trainieren
        for feld, neuron in enumerate(self.hidden_layer_1):
            for i, weight in enumerate(neuron.weights_n):
                abgeleiteter_fehler = relativer_fehler_hidden_1[feld] * ableitung_aktivierungsfunktion_hidden(ausgerechnete_werte_hidden_layer_1[feld])  * gegebener_input[i]

                neuron.weights_n[i] += lr * abgeleiteter_fehler # Gewicht anpassen
            neuron.weight_bias += relativer_fehler_hidden_1[feld] * lr # Bias anpassen  
            
class Neuron:
    def __init__(self, bias, weights_neuronen):
        self.weight_bias = bias
        self.weights_n = weights_neuronen
        
    def __repr__(self):
        return f"{self.weights_n}, {self.weight_bias}"