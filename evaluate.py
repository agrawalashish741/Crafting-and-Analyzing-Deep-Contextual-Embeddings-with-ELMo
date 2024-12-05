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
def preprocess_ag_news(texts, labels, vocab_to_idx, max_length):
    dataset = []
    for text, label in zip(texts, labels):
        token_idxs = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in text.lower().split()]
        token_idxs = [vocab_to_idx['<start>']] + token_idxs + [vocab_to_idx['<end>']]
        token_idxs = token_idxs[:max_length]  
        dataset.append((torch.tensor(token_idxs, dtype=torch.long), label))
    return dataset

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab_to_idx['<PAD>'])
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

def evaluate_model(elmo_model, classifier, dataloader, device, is_training=True):
    elmo_model.eval() 
    classifier.eval()     
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            _, _, combined_layer1, combined_layer2, embeddings = elmo_model(texts)
            outputs = classifier(combined_layer1, combined_layer2, embeddings) 
            _, predicted = torch.max(outputs, 1)      
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    if is_training:
        print("Evaluation on Training Data:")
    else:
        print("Evaluation on Testing Data:")      
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions))

    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    return accuracy
max_length=50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load('bilstm.pt')
loaded_model.to(device)
loaded_classifier= torch.load('classifier.pt')
loaded_classifier.to(device)
ag_news_train = pd.read_csv('/kaggle/input/news-classification-dataset/train.csv')
ag_news_test = pd.read_csv('/kaggle/input/news-classification-dataset/test.csv')
ag_news_train['Class Index'] = ag_news_train['Class Index'] - 1
ag_news_test['Class Index'] = ag_news_test['Class Index'] - 1
vocab_to_idx = torch.load('vocab_to_idx.pth')
train_dataset = preprocess_ag_news(ag_news_train['Description'][:25000], ag_news_train['Class Index'][:25000], vocab_to_idx, max_length)
test_dataset = preprocess_ag_news(ag_news_test['Description'], ag_news_test['Class Index'], vocab_to_idx, max_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
train_accuracy = evaluate_model(loaded_model, loaded_classifier, train_loader, device, is_training=True)
test_accuracy = evaluate_model(loaded_model, loaded_classifier, test_loader, device, is_training=False)