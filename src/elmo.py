import torch
from torch.nn import Module as BaseModule
import torch.optim as Optimizer
import torch.nn as NeuralNetwork
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as ConfusionMatrix
from .Config import ELMOConfig, Structure, Constants
from bidict import bidict
from alive_progress import alive_bar, alive_it
import numpy as Numpy
import matplotlib.pyplot as Plot
import seaborn as Seaborn

def createDataPoints(sentence : tuple[str], mapping : bidict):
    return torch.tensor( [ mapping.get(word, mapping[Constants.unkToken]) for word in sentence ] )

class PredictionDataset(TorchDataset):
    def __init__(self, sentences : list[tuple], indexing : bidict ) -> None:
        _, sentences = zip(*sentences)
        print(f"Creating Dataset")
        self.dataPoints = [createDataPoints(sentence,indexing) for sentence in alive_it(sentences, force_tty = True)]
        print(f"Dataset Created for {len(self.dataPoints)} sentences.")
        self.indexing = indexing
        print(self.dataPoints[0])

    def __len__(self):
        return len(self.dataPoints)

    def __getitem__(self, index : int) -> torch.Tensor:
        return self.dataPoints[index]

def predictionCollate(batch : list[torch.Tensor]):
    return NeuralNetwork.utils.rnn.pad_sequence(batch, batch_first = True, padding_value = 0)

class ELMO(BaseModule):
    def __init__(self,
        mapping,
        ) -> None :
        
        super().__init__()
        deviceString = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {deviceString}.")
        self.device = torch.device(deviceString)
        
        self.Embeddings = NeuralNetwork.Embedding(len(mapping), Constants.EmbeddingSize)
        
        print(f"Vocab Size : {len(mapping)}")
        print(f"Creating Embedding Layer of size : {Constants.EmbeddingSize}")
        print(f"Creating LSTM layers of size : {ELMOConfig.hiddenStateSize}")
        self.predictionLayer = NeuralNetwork.Linear(Constants.EmbeddingSize, len(mapping))

        self.LSTM1Forward = NeuralNetwork.LSTM(
            input_size = Constants.EmbeddingSize, 
            hidden_size = ELMOConfig.hiddenStateSize, 
            bidirectional = False,
            batch_first = True,
            device = self.device
        )
        self.LSTM2Forward = NeuralNetwork.LSTM(
            input_size = ELMOConfig.hiddenStateSize, 
            hidden_size = ELMOConfig.hiddenStateSize, 
            bidirectional = False,
            batch_first = True,
            device = self.device
        )

        self.LSTM1Backward = NeuralNetwork.LSTM(
            input_size = Constants.EmbeddingSize,
            hidden_size = ELMOConfig.hiddenStateSize,
            bidirectional = False,
            batch_first = True,
            device = self.device
        )
        self.LSTM2Backward = NeuralNetwork.LSTM(
            input_size = ELMOConfig.hiddenStateSize,
            hidden_size = ELMOConfig.hiddenStateSize,
            bidirectional = False,
            batch_first = True,
            device = self.device
        )
        
        self.to(self.device)
        
    def getEmbeddings(self, sentence):
        e0 = self.Embeddings(sentence)
        
        fe1 = self.LSTM1Forward(e0)[0]
        fe2 = self.LSTM2Forward(fe1)[0]
        
        be1 = self.LSTM1Backward(e0.flip(dims = (1,2)))[0]
        be2 = self.LSTM2Backward(be1.flip(dims = (1,2)))[0]
        
        e1 = torch.cat( (fe1, be1), dim = 2)
        e2 = torch.cat( (fe2, be2), dim = 2)

        return e0, e1, e2

    def forward(self, sentence):
        _, _, out = self.getEmbeddings(sentence)
        out = out.view(-1, out.shape[2])
        out = self.predictionLayer(out)
        return out

    def getFrozenEmbeddings(self, semtence):
        with torch.no_grad():
            return self.getEmbeddings(semtence)

#     
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # class ELMODatasetClassification(TorchDataset):
# #     def __init__(self, sentences : list[tuple], indexing : bidict ) -> None:
# #         self.sentences = sentences
# #         self.indexing = indexing
# #     
# #     def __len__(self):
# #         return len(self.sentences)
# #
# #     def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
# #         sentence = self.sentences[index][1]
# #         dataPoint = createDataPoints(sentence, self.mapping)
# #         label = self.sentences[index][0]
# #
#         return dataPoint, label
#
# class ELMODatasetPrediction(TorchDataset):
#     def __init__(self, sentences : list[tuple], indexing : bidict ) -> None:
#         self.sentences = sentences
#         self.indexing = indexing
#
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, index : int) -> torch.Tensor:
#         sentence = self.sentences[index][1]
#         sentenceLength = len(sentence)
#         dataPoint = torch.stack([ torch.tensor( self.indexing.get(token, 3) ) for token in sentence] )
#         dataPoint = dataPoint.view(-1)
#         return dataPoint, sentenceLength
#
# class ELMO(BaseModule):
#     def __init__(self, 
#                  trainSentences   : list[tuple],
#                  testSentences    : list[tuple],
#                  vocabIndexing    : bidict, 
#                  classifierName   : str     = "ELMO_Classifier.pt",
#                  predictionName   : str     = "ELMO_Predictor.pt",
#                  numClasses       : int     = ELMOConfig.numClasses,
#                  embeddingSize    : int     = ELMOConfig.EmbeddingSize,
#                  hiddenSize       : int     = ELMOConfig.hiddenStateSize,
#                  classifierLayers : tuple   = ELMOConfig.classifierLayers,
#                  predictionLayers : tuple   = ELMOConfig.predictionLayers,
#                  ) -> None:
#         super().__init__()
#
#         self.trainSentences = trainSentences
#         self.testSentences = testSentences
#
#         self.deviceString = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = torch.device(self.deviceString)
#
#         self.hiddenStateSize = hiddenSize
#         
#         self.classifierLayers = classifierLayers
#         self.numClasses = numClasses
#         self.classifierName = classifierName
#
#         self.vocabIndexing = vocabIndexing
#         self.vocabLength = len(self.vocabIndexing)
#         
#         self.EmbeddingSize = embeddingSize
#
#         self.predictionLayers = predictionLayers
#         self.predictionName = predictionName
#         
#         self.trainSetPrediction = ELMODatasetPrediction(self.trainSentences, indexing=self.vocabIndexing)
#         self.trainDatasetPrediction = DataLoader(
#             dataset=self.trainSetPrediction, 
#             batch_size=ELMOConfig.batchSize, 
#             collate_fn=self.customCollatePrediction, 
#             shuffle=True,
#             num_workers=ELMOConfig.numWorkers,
#             pin_memory=True
#         )
#         
#         self.testSetPrediction  = ELMODatasetPrediction(self.testSentences, indexing=self.vocabIndexing)
#         self.testDatasetPrediction = DataLoader(self.testSetPrediction, batch_size=ELMOConfig.batchSize, collate_fn=self.customCollatePrediction)
#         
#         self.trainDatasetPrediction = self.testDatasetPrediction
#
#         ############################# Embedding ############################
#         self.embedding = NeuralNetwork.Embedding( 
#             num_embeddings = self.vocabLength, 
#             embedding_dim = self.EmbeddingSize,
#         )
#
#         ######################### BI-LSTM Structure #########################
#
#         ### PreTrainedBiLSTM : Trained on the Next Word Prediction Task.
#         self.PreTrainedBiLSTMLayer1 = NeuralNetwork.LSTM(
#             input_size      = self.EmbeddingSize,
#             hidden_size     = self.hiddenStateSize,
#             num_layers      = ELMOConfig.numLayers,
#             batch_first     = True,
#             bidirectional   = ELMOConfig.bidirectional,
#             device          = self.device,
#             dropout         = ELMOConfig.dropout
#         )
#
#         ### PreTrainedBiLSTM : Trained on the Next Word Prediction Task.
#
#         self.PreTrainedBiLSTMLayer2 = NeuralNetwork.LSTM(
#             input_size      = self.hiddenStateSize*(1 + ELMOConfig.bidirectional),
#             hidden_size     = self.hiddenStateSize,
#             num_layers      = ELMOConfig.numLayers,
#             batch_first     = True,
#             bidirectional   = ELMOConfig.bidirectional,
#             device          = self.device,
#             dropout         = ELMOConfig.dropout
#         )
#
#         ### DownStreamBiLSTM : Trained on the Sentence Classification Task.
#         # self.DownStreamBiLSTM = NeuralNetwork.LSTM(
#         #     input_size      = self.hiddenStateSize,
#         #     hidden_size     = self.hiddenStateSize,
#         #     num_layers      = ELMOConfig.numLayers,
#         #     batch_first     = True,
#         #     bidirectional   = ELMOConfig.bidirectional,
#         #     device          = self.device,
#         #     dropout         = ELMOConfig.dropout
#         # )
#         
#         ######################### Classification #########################
#         # self.classifier = NeuralNetwork.ModuleList()
#         # for i in range(len(self.classifierLayers) + 1):
#         #     if i == len(self.classifierLayers) :
#         #         self.classifier.append(NeuralNetwork.Linear(in_features = self.classifierLayers[-1], out_features = self.numClasses))
#         #     else :
#         #         layerSize = self.classifierLayers[i]
#         #         if i == 0 :
#         #             self.classifier.append(NeuralNetwork.Linear(in_features = self.hiddenStateSize*(1 + ELMOConfig.bidirectional), out_features = layerSize))
#         #         else :
#         #             previousLayerSize = self.classifierLayers[i - 1]
#         #             self.classifier.append(NeuralNetwork.Linear(in_features = previousLayerSize, out_features = layerSize))
#         
#         ########################### Prediction ###########################
#         self.prediction = NeuralNetwork.ModuleList()
#         for i in range(len(self.predictionLayers) + 1):
#             if i == len(self.predictionLayers):
#                 if i == 0 :
#                     self.prediction.append(NeuralNetwork.Linear(in_features = self.hiddenStateSize*(1 + ELMOConfig.bidirectional), 
#                                                                 out_features = self.vocabLength))
#                 else :
#                     self.prediction.append(NeuralNetwork.Linear(in_features = self.predictionLayers[-1], out_features = self.vocabLength))
#             else :
#                 layerSize = self.predictionLayers[i]
#                 if i == 0 :
#                     self.prediction.append(NeuralNetwork.Linear(in_features = self.hiddenStateSize*(1 + ELMOConfig.bidirectional), 
#                                                                 out_features = layerSize))
#                 else :
#                     previousLayerSize = self.predictionLayers[i - 1]
#                     self.prediction.append(NeuralNetwork.Linear(in_features = previousLayerSize, out_features = layerSize))
#         
#         self.RELU = NeuralNetwork.ReLU()
#         
#         # self.classificationLossFunction = NeuralNetwork.CrossEntropyLoss(ignore_index = 0)
#         self.predictionLossFunction = NeuralNetwork.CrossEntropyLoss(ignore_index = 0)
#
#         # self.classificationOptimizer = Optimizer.Adam(self.parameters(), lr = ELMOConfig.classifierLearningRate)
#         self.predictionOptimizer = Optimizer.Adam(self.parameters(), lr = ELMOConfig.predictionLearningRate)
#
#         print(f"Using {self.deviceString} as device.")
#         self.trained = False
#         self.to(self.device)
#     
#     def saveModel(self, fileName):
#         filePath = Structure.modelPath + fileName
#         torch.save(self.state_dict(), filePath)
#         print(f"Saved model to {filePath}.")
#
#     # def __classifierForward(self, sentence):
#     #     _, (input, _) = self.ClassifierLSTM(sentence)
#     #     input = input.permute(1, 0, 2)
#     #     x = input.reshape(input.shape[0], -1)
#     #     for layer in self.classifier:
#     #         x = layer(x)
#     #         if layer != self.classifier[-1]:
#     #             x = self.RELU(x)
#     #         else :
#     #             x = self.softmax(x)
#     #
#     #     return x
#     #
#     # def trainClassifierStep(self, input, label):
#     #     self.classificationOptimizer.zero_grad()
#     #     y_hat = self.__classifierForward(input)
#     #     loss = self.classificationLossFunction(y_hat, label)
#     #     loss.backward()
#     #     self.classificationOptimizer.step()
#     #     return loss.item()
#     #
#     # def trainClassifier(self):
#     #     bestTestAccuracy = 80
#     #     for epoch in range(ELMOConfig.epochs):
#     #         avgLoss = 0
#     #         for x, y in self.trainDatasetClassification:
#     #             x = x.to(self.device)
#     #             y = y.to(self.device)
#     #             loss = self.trainClassifierStep(x,y)black suit
#     #             avgLoss += loss
#     #         avgLoss = avgLoss / len(self.trainDatasetClassification)
#     #         testAccuracy = self.evaluation()
#     #         print(f"Epoch : {epoch+1} | Loss: {avgLoss} | Test Accuracy : {testAccuracy}")
#     #         if testAccuracy > bestTestAccuracy :
#     #             bestTestAccuracy = testAccuracy
#     #             self.saveModel(f"{self.classifierName}_{str(testAccuracy)}.model")
#     # 
#     #
#     # def classifierEvaluation(self, data = None):
#     #     if data is None :
#     #         data = self.testData
#     #
#     #     correct = 0
#     #     predicted = []
#     #     actual = []
#     #
#     #     with torch.no_grad():
#     #         for y,x in data:
#     #             x = x.squeeze(2)
#     #             x = x.to(self.device)
#     #             y = y.to(self.device)
#     #             y_hat = self.__classifierForward(x)
#     #             y_hat = torch.argmax(y_hat)
#     #             y = torch.argmax(y)
#     #             predicted.append(y_hat.cpu().detach().numpy())
#     #             actual.append(y.cpu().detach().numpy())
#     #             correct += (y_hat == y)
#     #
#     #     self.confusionMatrix = ConfusionMatrix(actual, predicted)
#     #     return float(correct*100/ len(data))
#     
#     def __predictionF(self, x):
#         forwardHiddenStates  = x[:, :-1, :ELMOConfig.hiddenStateSize]
#         backwardHiddenStates = x[:, 1:, ELMOConfig.hiddenStateSize:]
#         
#         shape = x.shape[0],1,ELMOConfig.hiddenStateSize
#
#         forwardStatesAdjusted = torch.cat((torch.zeros(shape).to(self.device), forwardHiddenStates), dim = 1)
#         backwardStatesAdjusted = torch.cat((backwardHiddenStates, torch.zeros(shape).to(self.device)), dim = 1)
#
#         x = torch.cat((forwardStatesAdjusted, backwardStatesAdjusted), dim = 2)
#
#         return x
#
#
#     def __predictionForward(self, sentence, sentenceLengths):
#         sentenceLengths = list(sentenceLengths)
#         sentence = sentence.to(self.device)
#         sentence = self.embedding(sentence)
#         # sentence = NeuralNetwork.utils.rnn.pack_padded_sequence(sentence, sentenceLengths, batch_first = True, enforce_sorted = False)
#         out, _ = self.PreTrainedBiLSTMLayer1(sentence)
#         x = self.__predictionF(out)
#         out, _ = self.PreTrainedBiLSTMLayer2(x)
#         x = self.__predictionF(out)
#         for layer in self.prediction:
#             x = layer(x)
#             x = self.RELU(x)
#         
#         return x
#
#     def trainPredictionStep(self, input, inputLength):
#         self.predictionOptimizer.zero_grad()
#         yHat = self.__predictionForward(input, inputLength)
#
#         yHat = yHat.reshape(-1, yHat.shape[-1])
#         input = input.reshape(-1).to(self.device)
#         loss = self.predictionLossFunction(yHat, input)
#         loss.backward()
#         self.predictionOptimizer.step()
#         return loss.item()
#
#     def trainPrediction(self):
#         bestTestAccuracy = 80
#         for epoch in range(ELMOConfig.epochs):
#             avgLoss = 0
#             with alive_bar(len(self.trainDatasetPrediction), force_tty=True) as bar:
#                 for sentence, length in self.trainDatasetPrediction:
#                     loss = self.trainPredictionStep(sentence, length)
#                     avgLoss += loss
#                     bar()
#                 avgLoss = avgLoss / len(self.trainDatasetPrediction)
#                 testAccuracy = self.predictionEvaluation()
#                 print(f"Epoch : {epoch+1} | Loss:  {avgLoss} | Test Accuracy : {testAccuracy}")
#                 if testAccuracy > bestTestAccuracy :
#                     bestTestAccuracy = testAccuracy
#                     self.saveModel(f"{self.predictionName}_{str(testAccuracy).split('.')[0]}.model")
#
#     def predictionEvaluation(self, data=None):
#         if data is None:
#             data = self.testDatasetPrediction
#
#         correct = 0
#         total = 0
#
#         with torch.no_grad():
#             for sentences, lengths in data:
#                 batch_size = sentences.size(0)
#
#                 predictions = self.__predictionForward(sentences, lengths)
#                 
#                 # print(f"Predictions : {predictions.shape} | Labels : {labels.shape}")
#                 # Remove padding tokens based on the non_padding_mask
#                 for i in range(batch_size):
#                     seq_length = lengths[i]  # Calculate the actual sequence length
#                     prediction = predictions[i, :seq_length, :].argmax(dim=1).cpu().detach()
#                     label = sentences[i, :seq_length].flatten()
#                     correct += torch.sum(prediction == label).item()
#                     total += seq_length
#                     assert(len(prediction) == len(label) and len(prediction) == seq_length)
#                     assert(correct <= total)
#
#         accuracy = (correct / total) * 100
#         assert(accuracy <= 100)
#         return accuracy
#
#     # def customCollateClassification(self, batch):
#     #     # Extract sequences and labels from the batch
#     #     labels, sequences = zip(*batch)
#     #     # Pad sequences to the length of the longest sequence
#     #     paddedSequences = NeuralNetwork.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
#     #     # Convert labels to tensor
#     #     paddedLabels = NeuralNetwork.utils.rnn.pad_sequence(labels, batch_first=True)
#     #     return paddedSequences, paddedLabels
#
#     def customCollatePrediction(self, batch):
#         # Extract sequences and labels from the batch
#         sequences, lengths = zip(*batch)
#         # Pad sequences to the length of the longest sequence
#         paddedSequences = NeuralNetwork.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
#
#         # print(paddedSequences.shape)
#         # print(paddedLabels.shape)
#         return paddedSequences, lengths
