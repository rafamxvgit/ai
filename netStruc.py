import numpy as np

class net():
    """classe que define os objetos que representam as redes neurais do programa"""
    def _init__(self, *layersSizes):
        self.NUMLAYERS = len(layersSizes)
        self.LAYERSSIZES = layersSizes
        self.NUMCONECTIONS = (self.NUMLAYERS-1)

        self.weights = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
        self.offSets = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
    
    def setRandom(self, rangeWeights, rangeOffs = None):
        for layerX in range(1, self.NUMLAYERS):
            pass