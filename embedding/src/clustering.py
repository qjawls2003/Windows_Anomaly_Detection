from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


class Cluster:

    def __init__(self, dataset):
        self.dataset = dataset
        self.vectorizer = CountVectorizer()

    def process_data(self):
        pass
   