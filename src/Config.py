from pathlib import Path
import os

class Structure:
    """
        Information about the project structure, file management.
    """
    # Corpus is in the parent directory.
    corpusPath = os.path.join(Path(__file__).absolute().parents[1], "corpus/")
    
    # Path to save model weights
    modelPath = os.path.join(Path(__file__).absolute().parents[1], "pretrained/")

    # Train Path : corpus/train.csv
    trainPath = str(os.path.join(corpusPath, "train.csv"))

    # Test Path : corpus/test.csv
    testPath = str(os.path.join(corpusPath, "test.csv"))
    
    # results path 
    resultsPath = os.path.join(Path(__file__).absolute().parents[1], "results/")

class Constants:
    # Custom tokens
    startToken = "<s>"
    endToken = "</s>"
    unkToken = "<unk>"
    customTokens = {startToken, endToken, unkToken}

    cleanser = r'(\\+|\/\/+)'
    
    EmbeddingSize = 256
    
    # Context window
    contextWindow = 3

class SVDConfig(Constants):
    # NUmber of words to use for stochastic SVD.
    numWords = 1000
    

class Word2VecConfig(Constants):
    # Negative Samples
    negativeSamples = 2

    #number of epochs
    epochs = 10

    #batch size
    batchSize = 2**7
    
class ELMOConfig(Constants):
    # numLayers
    numLayers = 2
    
    # HiddenStateSize
    hiddenStateSize = 128

    # dropout
    dropout = 0.1 

    # bidirectional
    bidirectional = True
    
    # classifier layer
    classifierLayers = (64, 64, 32)
    
    # numClasses 
    numClasses = 4

    # prediction layer 
    predictionLayers = (64, 32, 32)

    # number of epochs
    epochs = 10
    
    # batch size
    batchSize = 2**6
    
    # classifier learning rate
    classifierLearningRate = 0.001

    # prediction learning rate
    predictionLearningRate = 0.001
    
class ClassificationConfig(Constants):
    # Embeddding Size
    HiddenStateSize = 128
    
    # learning rate
    learningRate = 0.001

    # numClasses 
    numClasses = 4

    # bidirectional
    bidirectional = True
    
    # batch size 
    batchSize = 16    
    # epochs
    epochs = 8
