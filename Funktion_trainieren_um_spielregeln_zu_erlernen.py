def trainieren(self, bewertung, ausgerechneter_outp_und_hidden_layer, gegebener_input, aktivierungsfunktion_hidden, aktivierungsfunktion_outp):
#Die Funktion trainieren um das Neuronale Netz so zu trainieren dass es nur noch legale Züge spielt. Dabei ignorriert man ob die Ki gewonnen oder verloren hat.

    ausgerechneter_outp = []
    ausgerechneter_hidden_layer_1 = []
    ausgerechneter_hidden_layer_2 = []
    gespielte_züge = [] #auf welches feld wurde gespielt
    
    for züge in ausgerechneter_outp_und_hidden_layer:
        ausgerechneter_outp.append(züge[0])
        ausgerechneter_hidden_layer_1.append(züge[1])
        ausgerechneter_hidden_layer_2.append(züge[2])
        gespielte_züge.append(züge[0].index(max(züge[0])))

        
    if bewertung == -100:
        letzter_zug_feld = gespielte_züge[-1]
        aktueller_letzter_zug = gespielte_züge.index(letzter_zug_feld) #der wievielte zug war der letzte?
        optimalen_outp =[]
        for i in range(self.anzahl_output):
            optimalen_outp.append(ausgerechneter_outp[aktueller_letzter_zug][i] * 1.01) #sagt das der ausgerechnete output = dem gespielten output ist. könte auch
                
        optimalen_outp[letzter_zug_feld] = ausgerechneter_outp[aktueller_letzter_zug][letzter_zug_feld] * 0.75         
        self.backpropagation(optimalen_outp, ausgerechneter_outp[aktueller_letzter_zug], ausgerechneter_hidden_layer_1[aktueller_letzter_zug], ausgerechneter_hidden_layer_2[aktueller_letzter_zug], gegebener_input[aktueller_letzter_zug], aktivierungsfunktion_hidden, aktivierungsfunktion_outp)
        
        
        for aktueller_zug, zug_feld in enumerate(gespielte_züge[0:(len(gespielte_züge)-1)]): #ohne den letzten zug
            optimalen_outp =[]
            for i in range(self.anzahl_output):
                optimalen_outp.append(ausgerechneter_outp[aktueller_zug][i] * 0.98)
                
            optimalen_outp[zug_feld] = ausgerechneter_outp[aktueller_zug][zug_feld] * 1.25
            self.backpropagation(optimalen_outp, ausgerechneter_outp[aktueller_zug], ausgerechneter_hidden_layer_1[aktueller_zug], ausgerechneter_hidden_layer_2[aktueller_zug], gegebener_input[aktueller_zug], aktivierungsfunktion_hidden, aktivierungsfunktion_outp)
    
    
    if bewertung == 1 or bewertung == -1 or bewertung == 100 or bewertung == 0: #alle bewertungen werden gleich behandelt, weil alle züge legal waren. 
        for aktueller_zug, zug_feld in enumerate(gespielte_züge):
            optimaler_outp =[]
            for i in range(self.anzahl_output):
                optimaler_outp.append(ausgerechneter_outp[aktueller_zug][i] * 0.9)
            optimaler_outp[zug_feld] = ausgerechneter_outp[aktueller_zug][zug_feld] * 1.25
            self.backpropagation(optimaler_outp, ausgerechneter_outp[aktueller_zug], ausgerechneter_hidden_layer_1[aktueller_zug], ausgerechneter_hidden_layer_2[aktueller_zug], gegebener_input[aktueller_zug], aktivierungsfunktion_hidden, aktivierungsfunktion_outp)
        return
