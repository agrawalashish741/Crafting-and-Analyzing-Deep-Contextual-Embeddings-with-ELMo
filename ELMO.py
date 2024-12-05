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

model_path = '/kaggle/input/word2vec-google-3-dimensions/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = pd.read_csv('/kaggle/input/news-classification-dataset/train.csv')
all_sentences = list(train_data['Description'])
word_freq = Counter(word for sentence in all_sentences for word in sentence.split())
vocab = {word for word, freq in word_freq.items() if freq >= 3}
special_tokens = ['<PAD>', '<start>', '<end>', '<UNK>']
vocab.update(special_tokens)
vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab_to_idx)  
class TextDataset(Dataset):
    def __init__(self, sentences, vocab_to_idx, max_length):
        self.vocab_to_idx = vocab_to_idx
        self.data = [self.preprocess_text(sentence, max_length) for sentence in sentences]

    def preprocess_text(self, text, max_length):
        tokens = text.lower().split()
        token_idxs = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        token_idxs = [self.vocab_to_idx['<start>']] + token_idxs + [self.vocab_to_idx['<end>']]
        token_idxs = token_idxs[:max_length]
        input_idxs = torch.tensor(token_idxs[:-1], dtype=torch.long)
        target_idxs = torch.tensor(token_idxs[1:], dtype=torch.long)
        return input_idxs, target_idxs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        pad_idx = vocab_to_idx['<PAD>']
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        return inputs_padded, targets_padded

max_length = 50
dataset = TextDataset(all_sentences, vocab_to_idx, max_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=TextDataset.collate_fn)
class ELMoModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, dropout=0.5):
        super(ELMoModel, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
        self.forward_lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.forward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.backward_lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.backward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_forward = nn.Linear(hidden_dim, vocab_size)
        self.output_backward = nn.Linear(hidden_dim, vocab_size)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.combined_output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embeddings = self.embedding_projection(self.dropout(self.embedding(x)))
        forward_out1, _ = self.forward_lstm1(embeddings)
        forward_out2, _ = self.forward_lstm2(forward_out1)
        
        backward_embeddings = self.embedding_projection(self.dropout(self.embedding(torch.flip(x, [1]))))
        backward_out1, _ = self.backward_lstm1(backward_embeddings)
        backward_out2, _ = self.backward_lstm2(backward_out1)
        
        forward_predictions = self.output_forward(forward_out2) 
        backward_predictions = self.output_backward(backward_out2)  
        combined_layer1 = torch.cat((forward_out1, backward_out1), dim=-1)
        combined_layer2 = torch.cat((forward_out2, backward_out2), dim=-1)
        combined_layer1 = self.combined_output_projection(combined_layer1)
        combined_layer2 = self.combined_output_projection(combined_layer2)

     

        return forward_predictions, backward_predictions, combined_layer1 , combined_layer2,embeddings

embedding_matrix = torch.zeros(vocab_size, 300)
for word, idx in vocab_to_idx.items():
    if word in word2vec_model:
        embedding_matrix[idx] = torch.tensor(word2vec_model[word], dtype=torch.float32)
    else:
        embedding_matrix[idx] = torch.rand(300)

model = ELMoModel(vocab_size, 300, 300, embedding_matrix).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab_to_idx['<PAD>']).to(device)

def train(model, dataloader, optimizer, criterion, epochs=5, device=device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            forward_pred, backward_pred, _ ,_,_= model(inputs)
            loss_forward = criterion(forward_pred.view(-1, vocab_size), targets.view(-1))
            loss_backward = criterion(backward_pred.view(-1, vocab_size), targets.view(-1))
            loss = (loss_forward + loss_backward) / 2
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}')

train(model, dataloader, optimizer, criterion)
torch.save(model, 'bilstm.pt')
torch.save(vocab_to_idx, 'vocab_to_idx.pth')