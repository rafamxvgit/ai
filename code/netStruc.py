import numpy as np

class net():
    """classe que define os objetos que representam as redes neurais do programa"""
    def __init__(self, *layersSizes):
        
        self.NUMLAYERS = len(layersSizes)
        if (self.NUMLAYERS < 2): 
            raise ValueError("Uma rede neural precisa ter pelo menos duas layers. NUMLAYERS=" + str(self.NUMLAYERS))

        self.LAYERSSIZES = layersSizes
        self.NUMCONECTIONS = self.NUMLAYERS-1

        self.weights = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
        self.offSets = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
    
    def setRandom(self, rangeWeights=(0,1), rangeOffs=(0,1)):
        '''A função "setRandom" serve para setar os pesos e os offsets da rede neural de forma aleatória.\n
        rangeWeights: tuple de dois números indicando o intervalo de valores que os pesos podem assumir. (low, high)\n
        rangeOffs: tuple de dois números indicando o intervalo de valores que os offsets podem assumir. (low, high)'''

        for layerX in range(1, self.NUMLAYERS):
            weighShape = (self.LAYERSSIZES[layerX], self.LAYERSSIZES[layerX-1])
            self.weights[layerX-1] = np.random.uniform(rangeWeights[0], rangeWeights[1], weighShape)
            self.offSets[layerX-1] = np.random.uniform(rangeOffs[0], rangeOffs[1], (self.LAYERSSIZES[layerX], 1))