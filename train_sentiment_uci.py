import copy
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from micrograd.loss import BinaryCrossEntropyLoss
from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.optimizer import SGD
from micrograd.functional import sigmoid
from micrograd.dataset import UCISentimentDataset
from micrograd.dataloader import DataLoader

LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 64
DATASET_ROOT = '/home/minh/Desktop/datasets/'
WEIGHTS_PATH = "sentiment_mlp.npz"
VECTORIZER_PATH = "sentiment_vectorizer.pkl"
LOG_ROOT = "./logs/sentiment"


if __name__ == '__main__':
    vectorizer = CountVectorizer()    
    ds = UCISentimentDataset(root=DATASET_ROOT, vectorizer=vectorizer)
    
    train, val = train_test_split(ds, test_size=0.2, random_state=42)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False)
    
    input_size = len(ds.vectorizer.vocabulary_)
    model = MLP(nin=input_size, nouts=[16, 1])
    criterion = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), learning_rate=LEARNING_RATE)
    
    logs = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': []
    }
    
    best_val_acc = float('-inf')
    best_model = None
    for epoch in range(EPOCHS):
        train_acc = 0.
        train_loss = 0.
        for inputs, labels in train_loader:
            inputs = Value(inputs)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            train_loss += loss.data
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_scores = sigmoid(logits)
            train_preds = (train_scores >= 0.5).astype(int)
            train_acc += np.mean(labels == train_preds)
            
            logs['train_loss'].append(loss.data)
            
        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        print(f"Epoch: {epoch + 1}. Train loss={train_loss}. Train acc={train_acc * 100:.2f}%")
        logs['train_acc'].append(train_acc)
        
        val_acc = 0.
        for inputs, labels in val_loader:
            inputs = Value(inputs)
            
            logits = model(inputs)            
            val_scores = sigmoid(logits)
            val_preds = (val_scores >= 0.5).astype(int)
            val_acc += np.mean(labels == val_preds)
            
        val_acc /= len(val_loader)
        print(f"Epoch: {epoch + 1}. Val acc={val_acc * 100:.2f}%")
        logs['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            
    best_model.save_weights(WEIGHTS_PATH)
    with open(VECTORIZER_PATH, 'wb') as fout:
        pickle.dump(vectorizer, fout)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    os.makedirs(LOG_ROOT, exist_ok=True)
    
    plt.plot(logs['train_acc'], label='Train accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training accuracy graph')
    plt.savefig(os.path.join(LOG_ROOT, 'train_accuracy.jpg'))
    plt.clf()
    
    plt.plot(logs['val_acc'], label='Val accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Val accuracy graph')
    plt.savefig(os.path.join(LOG_ROOT, 'val_accuracy.jpg'))
    plt.clf()
    
    plt.plot(logs['train_loss'], label='Train losses')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Training loss graph')
    plt.savefig(os.path.join(LOG_ROOT, 'loss.jpg'))
