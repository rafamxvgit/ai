import numpy as np
import time as t

class net():
    """classe que define os objetos que representam as redes neurais do programa"""
    def __init__(self, *layersSizes):
        
        self.NUMLAYERS = len(layersSizes)
        if (self.NUMLAYERS < 2): 
            raise ValueError("Uma rede neural precisa ter pelo menos duas layers. NUMLAYERS=" + str(self.NUMLAYERS))

        self.LAYERSSIZES = layersSizes # a quantidade de neurônios em cada layer
        self.NUMCONECTIONS = self.NUMLAYERS-1

        self.weights = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
        self.offSets = [None]*self.NUMCONECTIONS #colocar somente arrays de numpy aqui dentro
    
    def setRandom(self, rangeWeights=(0,1), rangeOffs=(0,1)):
        '''A função "setRandom" serve para setar os pesos e os offsets (todas as conecções) da rede neural de forma aleatória.\n
        rangeWeights: tuple de dois números indicando o intervalo de valores que os pesos podem assumir. (low, high)\n
        rangeOffs: tuple de dois números indicando o intervalo de valores que os offsets podem assumir. (low, high)'''

        for layerX in range(1, self.NUMLAYERS):
            weighShape = (self.LAYERSSIZES[layerX], self.LAYERSSIZES[layerX-1])
            self.weights[layerX-1] = np.random.uniform(rangeWeights[0], rangeWeights[1], weighShape)
            self.offSets[layerX-1] = np.random.uniform(rangeOffs[0], rangeOffs[1], (self.LAYERSSIZES[layerX], 1))

    def run(self, inputLayer, complete = False):
        """A função retorna a "resposta" da rede neural a um dado input"""
        inputErrorMessage = """A inputLayer deve ter tamanho compativel cor a primeira layer da rede\nSize expected: {}\nSize passed: {}"""
        inputShape = inputLayer.shape
        shapeExpected = (self.LAYERSSIZES[0], 1)
        if (inputShape != (shapeExpected)): raise ValueError(inputErrorMessage.format(shapeExpected, inputShape))
        
        sigmoid = lambda x: 1/(1+np.exp(x))
        
        if complete == True:
            layers = [inputLayer]
            for x in range(self.NUMCONECTIONS):
                newLayer = sigmoid(np.matmul(self.weights[x], layers[-1])+self.offSets[x])
                layers.append(newLayer)
            return(layers)
        
        else:
            outputLayer = inputLayer
            for x in range(self.NUMCONECTIONS):
                outputLayer = sigmoid(np.matmul(self.weights[x], outputLayer)+self.offSets[x])
            return(outputLayer)

    def cost(self, inputLayer, expectedOutputLayer):
        netOutput = self.run(inputLayer)
        if (expectedOutputLayer.shape != netOutput.shape): raise ValueError("tamanho de output incompatível com a rede neural")
        return(np.sum(np.power(expectedOutputLayer-netOutput, 2)))
