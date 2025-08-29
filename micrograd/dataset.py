import os
import gzip
import zipfile
import shutil
import urllib.request
import numpy as np
from glob import glob


class MNISTDataset:
    @staticmethod
    def load_mnist_img(path):
        try:
            with open(path, "rb") as fi:
                _ = int.from_bytes(fi.read(4), "big")  # magic number
                n_images = int.from_bytes(fi.read(4), "big")
                h = int.from_bytes(fi.read(4), "big")
                w = int.from_bytes(fi.read(4), "big")
                buffer = fi.read()
                images = np.frombuffer(buffer, dtype=np.uint8).reshape(n_images, h, w)
        except Exception as e:
            print(f"Could not read MNIST image file at {path}")
            print(e)
            exit(1)
        return images
    
    @staticmethod
    def load_mnist_lbl(path):
        try:
            with open(path, "rb") as fi:
                _ = int.from_bytes(fi.read(4), "big")
                n_labels = int.from_bytes(fi.read(4), "big")
                buffer = fi.read()
                labels = np.frombuffer(buffer, dtype=np.uint8)
        except Exception as e:
            print(f"Could not read MNIST label file at {path}")
            print(e)
            exit(1) 
        return labels
    
    @staticmethod
    def _download_and_extract(root):
        """
        Downloads and extracts the MNIST dataset files if they don't exist.
        """
        mnist_path = os.path.join(root, "MNIST")
        os.makedirs(mnist_path, exist_ok=True)
        
        urls = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
        ]

        for url in urls:
            filename = url.split("/")[-1]
            gz_path = os.path.join(mnist_path, filename)
            uncompressed_path = os.path.join(mnist_path, filename[:-3])

            if not os.path.exists(uncompressed_path):
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, gz_path)

                print(f"Extracting {gz_path}")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(uncompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
    
    '''
    dataset_dir
    |---MNIST
        ├── train-images.idx3-ubyte (train images file)
        ├── train-labels.idx1-ubyte
        ├── t10k-images.idx3-ubyte (val images file)
        ├── t10k-labels.idx1-ubyte
    '''
    
    def __init__(self, root, download=True, train=True):
        if download and not os.path.exists(os.path.join(root, "MNIST")):
            self._download_and_extract(root)
        
        if train:
            img_dir = os.path.join(root, "MNIST", "train-images-idx3-ubyte")
            lbl_dir = os.path.join(root, "MNIST", "train-labels-idx1-ubyte")
        else:
            img_dir = os.path.join(root, "MNIST", "t10k-images-idx3-ubyte")
            lbl_dir = os.path.join(root, "MNIST", "t10k-labels-idx1-ubyte")
        
        images = self.load_mnist_img(img_dir)
        labels = self.load_mnist_lbl(lbl_dir)
        
        self.data = [(image, label) for image, label in zip(images, labels)]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    
class UCISentimentDataset:
    """
    dataset_dir
    |---uci_sentiment
        ├── yelp_labelled.txt           (reviews from yelp, labelled with positive/negative sentiment)
        ├── amazon_cells_labelled.txt   (reviews from amamzon, labelled with positive/negative sentiment)
        ├── imdb_labelled.txt           (reviews from imdb, labelled with positive/negative sentiment)
    """
    
    @staticmethod
    def _download_and_extract(root):
        dataset_path = os.path.join(root, "uci_sentiment")
        os.makedirs(dataset_path, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
        filename = url.split("/")[-1]
        z_path = os.path.join(dataset_path, filename)
        uncompressed_path = os.path.join(dataset_path, filename[:-4])
        
        if not os.path.exists(uncompressed_path):
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, z_path)

            print(f"Extracting {z_path}")
            with zipfile.ZipFile(z_path, 'r') as f_in:
                f_in.extractall(uncompressed_path)
            os.remove(z_path)
            
            txt_paths = glob(uncompressed_path + f'/**/*.txt', recursive=True)
            for path in txt_paths:
                shutil.move(path, os.path.join(dataset_path, os.path.basename(path)))
            shutil.rmtree(uncompressed_path)
    
    def __init__(self, root, vectorizer, download=True):
        if download and not os.path.exists(os.path.join(root, "uci_sentiment")):
            self._download_and_extract(root)
        
        yelp_path = os.path.join(root, "uci_sentiment", 'yelp_labelled.txt')
        amazon_path = os.path.join(root, "uci_sentiment", 'amazon_cells_labelled.txt')
        imdb_path = os.path.join(root, "uci_sentiment", 'imdb_labelled.txt')

        self.sentences = []
        self.labels = []
        for path in [yelp_path, amazon_path, imdb_path]:
            with open(path, 'r') as f:
                for line in f:
                    sentence, label = line.strip().split('\t')
                    self.labels.append(int(label))
                    self.sentences.append(sentence)
        
        self.vectorizer = vectorizer
        self.vectorizer.fit(self.sentences)
        self.samples = self.vectorizer.transform(self.sentences).toarray()
        self.samples = [inp.flatten().astype(np.float32) for inp in self.samples]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = np.array([self.labels[item]])
        return sample, label
