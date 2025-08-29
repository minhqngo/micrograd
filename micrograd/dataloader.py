import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[j] for j in batch_indices]
            inputs, labels = zip(*batch_data)
            inputs_batch = np.array(inputs)
            labels_batch = np.array(labels)
            
            yield inputs_batch, labels_batch    
