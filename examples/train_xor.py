import os
import numpy as np
import matplotlib.pyplot as plt

from micrograd.loss import MSELoss
from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.optimizer import SGD
from micrograd.functional import softmax
from micrograd.dataloader import DataLoader

LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 1
WEIGHTS_PATH = "../xor_mlp.npz"
LOG_ROOT = "../logs/xor"


if __name__ == '__main__':
    train_ds = [
        (np.array([0, 0]), 0),
        (np.array([1, 0]), 1),
        (np.array([0, 1]), 1),
        (np.array([1, 1]), 0)
    ]
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

    model = MLP(nin=2, nouts=[16, 2])
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=LEARNING_RATE)

    logs = {
        'train_acc': [],
        'train_loss': []
    }

    for epoch in range(EPOCHS):
        train_acc = 0.
        train_loss = 0.
        for inputs, labels in train_loader:
            logits = model(Value(inputs))
            loss = criterion(logits, Value(labels))
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

    model.save_weights(WEIGHTS_PATH)

    os.makedirs(LOG_ROOT, exist_ok=True)

    plt.plot(logs['train_acc'], label='Train accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training accuracy graph')
    plt.savefig(os.path.join(LOG_ROOT, 'train_accuracy.jpg'))
    plt.clf()

    plt.plot(logs['train_loss'], label='Train losses')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Training loss graph')
    plt.savefig(os.path.join(LOG_ROOT, 'loss.jpg'))
