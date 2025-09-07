import copy
import os
import numpy as np
import matplotlib.pyplot as plt

from micrograd.loss import CrossEntropyLoss
from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.optimizer import SGD, NesterovSGD
from micrograd.functional import softmax
from micrograd.dataset import MNISTDataset
from micrograd.dataloader import DataLoader

LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 20
BATCH_SIZE = 128
DATASET_ROOT = "/home/minh/datasets/"
WEIGHTS_PATH = "../mnist_mlp.npz"
LOG_ROOT = "../logs/mnist"


def preprocess_inputs(inputs):
    flattened = inputs.flatten()
    flattened = flattened / 255.
    return flattened

    
if __name__ == '__main__':
    print("Loading dataset...")
    train_ds = MNISTDataset(root=DATASET_ROOT, train=True, download=True)
    val_ds = MNISTDataset(root=DATASET_ROOT, train=False, download=True)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    print("MNIST dataset loaded")
    
    # Input is 784 (28x28), one hidden layer of 32 neurons, one hidden layer of 16 neurons, output is 10 classes.
    model = MLP(nin=784, nouts=[32, 16, 10])
    criterion = CrossEntropyLoss()
    optimizer = NesterovSGD(model.parameters(), learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    
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
            inputs = preprocess_inputs(Value(inputs))
            
            logits = model(inputs)
            loss = criterion(logits, Value(labels, dtype=np.uint8))
            train_loss += loss.data
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_probs = softmax(logits)
            train_preds = np.argmax(train_probs, axis=1)
            train_acc += np.mean(labels == train_preds)
            
            logs['train_loss'].append(loss.data)
            
        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        print(f"Epoch: {epoch + 1}. Train loss={train_loss}. Train acc={train_acc * 100:.2f}%")
        logs['train_acc'].append(train_acc)
        
        val_acc = 0.
        for inputs, labels in val_loader:
            inputs = preprocess_inputs(Value(inputs))
            
            logits = model(inputs)            
            val_probs = softmax(logits)
            val_preds = np.argmax(val_probs, axis=1)
            val_acc += np.mean(labels == val_preds)
            
        val_acc /= len(val_loader)
        print(f"Epoch: {epoch + 1}. Val acc={val_acc * 100:.2f}%")
        logs['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            
    best_model.save_weights(WEIGHTS_PATH)
    
    os.makedirs(LOG_ROOT, exist_ok=True)
    
    plt.plot(logs['train_acc'], label='Train accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training accuracy graph')
    plt.savefig(os.path.join(LOG_ROOT, 'train_accuracy.jpg'))
    plt.clf()
    
    plt.plot(logs['val_acc'], label='Val accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Val accuracy graph')
    plt.savefig(os.path.join(LOG_ROOT, 'val_accuracy.jpg'))
    plt.clf()
    
    plt.plot(logs['train_loss'], label='Train losses')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Training loss graph')
    plt.savefig(os.path.join(LOG_ROOT, 'loss.jpg'))
