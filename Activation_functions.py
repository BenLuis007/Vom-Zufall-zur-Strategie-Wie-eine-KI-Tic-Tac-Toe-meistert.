import math

def sigmoid(x):
    if x < -100: #damit kein overflow entsteht
        x = -100
    if x > 100: #damit kein overflow entsteht
        x = 100
    return 1 / (1 + math.exp(-x))

def ableitung_sigmoid(x): # Die Abgeleitete Aktivierungsfunktion wird für Backpropagation gebraucht
    return x * (1-x)

def ReLU(x):
    if x > 0:
        return x
    else:
        return 0
def ableitung_ReLU(x): # Die Abgeleitete Aktivierungsfunktion wird für Backpropagation gebraucht
    if x > 0:
        return 1
    else:
        return 0

def linear(x):
    return x

def ableitung_linear(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0

def binary_step(x):
    if x > 0:
        return 1
    if x < 0:
        return 0

def leakyReLU(x):
    if x > 0:
        return x
    else:
        return x/10
