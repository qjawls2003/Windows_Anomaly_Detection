from gensim.models import Word2Vec


def vectorize(dataset):

    model = Word2Vec(dataset, min_count=1)

    #embedding = model.wv['Harambe', 'Downloads']

    return model