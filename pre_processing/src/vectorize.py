from gensim.models import Word2Vec
from csv_to_df import csv_to_df


def vectorize():

    dataset = csv_to_df()

    model = Word2Vec(train_commands, min_count=1, size=100)

    embedding = model.wv['Harambe', 'Downloads']

    print(embedding)