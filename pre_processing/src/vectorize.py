from transformers import BertTokenizer, BertModel

'''
BERT: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that learns contextualized word embeddings.
It considers the surrounding context of words to generate highly expressive representations.
 BERT is effective when logs require capturing fine-grained semantic relationships and context-specific meanings.
'''

def vectorize(dataset):

    model = Word2Vec(dataset, min_count=1)

    #embedding = model.wv['Harambe', 'Downloads']

    return model