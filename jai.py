import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import pandas as pd
import numpy as np
import pickle as pkl
from nltk.tokenize import word_tokenize
from bidict import bidict
from alive_progress import alive_bar, alive_it
import os
from ordered_set import OrderedSet

embeddingSize = 256
batchSize = 2**3
epochs = 10
lr = 1e-3

class ELMo(nn.Module):
    def __init__(self, inputSize, vocabSize):
        super(ELMo, self).__init__()
        self.emb = nn.Embedding(vocabSize, inputSize)
        self.bl1 = nn.LSTM(inputSize, inputSize // 2, 1, bidirectional=True, batch_first=True)
        self.bl2 = nn.LSTM(inputSize, inputSize // 2, 1, bidirectional=True, batch_first=True)

    def forward(self, x):
        e0 = self.emb(x)
        # e0 = pack_padded_sequence(e0, l, batch_first=True, enforce_sorted=False)
        e1, _ = self.bl1(e0)
        e2, _ = self.bl2(e1)
        # e2, _ = pad_packed_sequence(e2, batch_first=True)
        return e0, e1, e2
    
class PreTrainModel(ELMo):
    def __init__(self, inputSize, vocabSize):
        super(PreTrainModel, self).__init__(inputSize, vocabSize)
        self.fc = nn.Linear(inputSize, vocabSize)
        
    def forward(self, x):
        _, _, e2 = super(PreTrainModel, self).forward(x)
        o1 = torch.cat([torch.zeros(e2.shape[0], 1, e2.shape[2] // 2).to(e2.device),
                        e2[:, :-1, :e2.shape[2] // 2]], 1)
        o2 = torch.cat([e2[:, 1:, e2.shape[2] // 2:],
                        torch.zeros(e2.shape[0], 1, e2.shape[2] // 2).to(e2.device)], 1)
        output = torch.cat([o1, o2], 2)
        output = output.view(-1, output.shape[2])
        output = self.fc(output)
        return output
    
def collate_fn(batch, pad):
    data = pad_sequence(batch, batch_first=True, padding_value=pad)
    return data
    
def preTrainELMO(xTrain, xTest, batchSize, inputSize, epochs, lr, vocab, device):
    trainDL = DataLoader(xTrain, batch_size=batchSize, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab['<pad>']))
    
    model = PreTrainModel(inputSize, len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []
        with alive_bar(len(trainDL)) as bar:
            for x in trainDL:
                x = x.to(device)
                optimizer.zero_grad()
                output = model(x)
                x = x.view(-1)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar()
        print(f'Epoch {epoch+1}/{epochs} Loss: {np.mean(losses)}')
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for x in xTest:
                x = x.to(device).unsqueeze(0)
                output = model(x)
                x = x.view(-1)
                correct += torch.sum(torch.argmax(output, 1) == x).item()
                total += x.shape[0]
            print(f'Val Accuracy: {correct/total}')
            model.train()
    return model
    
if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(1.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    data = pd.read_csv('./data/train.csv')

    def removeBs(text):
        return text.replace('\\', ' ').replace('\n', ' ')

    def tokenize(text):
        tkList = ["<s>"]
        tkList += word_tokenize(text)
        tkList.append("</s>")
        return tkList

    data['Description'] = data['Description'].apply(removeBs)
    data['Description'] = data['Description'].apply(tokenize)

    train = data['Description']

    testData = pd.read_csv('./data/test.csv')
    testData['Description'] = testData['Description'].apply(removeBs)
    testData['Description'] = testData['Description'].apply(tokenize)
    test = testData['Description']

    if os.path.exists('./data/vocab.pkl'):
        with open('./data/vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)
    else:
        vocab = OrderedSet()
        with alive_bar(len(data)) as bar:
            for text in data['Description']:
                vocab.update(text)
                bar()
        vocab = bidict({word: idx for idx, word in enumerate(vocab)})
        vocab['<pad>'] = len(vocab)
        vocab['<unk>'] = len(vocab)
        with open('./data/vocab.pkl', 'wb') as f:
            pkl.dump(vocab, f)
    print("Vocab Size : ", len(vocab))

    def getIndex(word):
        try:
            return vocab[word]
        except:
            return vocab['<unk>']

    def getIndices(text):
        return torch.tensor([getIndex(word) for word in text]).long()

    xTrain = [getIndices(text) for text in alive_it(train)]
    xTest = [getIndices(text) for text in alive_it(test)]

    model = preTrainELMO(xTrain, xTest, batchSize, embeddingSize, epochs, lr, vocab, device)
    torch.save(super(PreTrainModel, model).state_dict(), './bilstm.pt')