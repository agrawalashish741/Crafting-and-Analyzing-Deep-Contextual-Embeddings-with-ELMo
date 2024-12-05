from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,classification_report
from gensim.models import KeyedVectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load('bilstm.pt')
loaded_model.to(device)
ag_news_train = pd.read_csv('/kaggle/input/news-classification-dataset/train.csv')
ag_news_test = pd.read_csv('/kaggle/input/news-classification-dataset/test.csv')
ag_news_train['Class Index'] = ag_news_train['Class Index'] - 1
ag_news_test['Class Index'] = ag_news_test['Class Index'] - 1
vocab_to_idx = torch.load('vocab_to_idx.pth')
max_length=50
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

freeze_model(loaded_model)

def preprocess_ag_news(texts, labels, vocab_to_idx, max_length):
    dataset = []
    for text, label in zip(texts, labels):
        token_idxs = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in text.lower().split()]
        token_idxs = [vocab_to_idx['<start>']] + token_idxs + [vocab_to_idx['<end>']]
        token_idxs = token_idxs[:max_length]  # Correct slicing using local max_length
        dataset.append((torch.tensor(token_idxs, dtype=torch.long), label))
    return dataset

train_dataset = preprocess_ag_news(ag_news_train['Description'][:25000], ag_news_train['Class Index'][:25000], vocab_to_idx, max_length)
test_dataset = preprocess_ag_news(ag_news_test['Description'], ag_news_test['Class Index'], vocab_to_idx, max_length)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab_to_idx['<PAD>'])
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, output_dim, num_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=300,  
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bidirectional=bidirectional, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_dim)
        self.lambdas = nn.Parameter(torch.randn(3, requires_grad=True))  

    def forward(self, combined_layer1, combined_layer2, embeddings):
        
        weights = torch.softmax(self.lambdas, dim=0)
        combined_embeddings = weights[0] * combined_layer1 + weights[1] * combined_layer2 + weights[2] * embeddings
        combined_embeddings = self.dropout(combined_embeddings)
        lstm_out, (hidden, _) = self.lstm(combined_embeddings)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        hidden = self.dropout(hidden)
        return self.fc(hidden)

hidden_size = 256  
output_dim = 4  
num_layers = 2
bidirectional = True
dropout = 0.5

classifier = LSTMClassifier(hidden_size, output_dim, num_layers, bidirectional, dropout).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)



def train_elmo_classifier(elmo_model, classifier, dataloader, optimizer, criterion, epochs=5,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_sentences=25000):
    classifier.to(device)
    elmo_model.to(device)
    classifier.train()
    elmo_model.eval()  

    for epoch in range(epochs):
        total_loss = 0
        total_sentences = 0  
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            batch_size = texts.size(0)  
            with torch.no_grad():
                _,_,combined_layer1 , combined_layer2,embeddings= elmo_model(texts)  
            optimizer.zero_grad()
            outputs = classifier(combined_layer1, combined_layer2, embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sentences += batch_size   
            if total_sentences >= max_sentences:
                break         
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}')

train_elmo_classifier(loaded_model, classifier, train_loader, optimizer, criterion, max_sentences=25000)
torch.save(classifier, 'classifier.pt')