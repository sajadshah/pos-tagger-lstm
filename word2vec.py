
from gensim.models import Word2Vec, Doc2Vec, LdaModel



def createW2VModel(data_iter, save_model_file = None):
    model = Word2Vec(size=100, window=5, min_count=5, workers=4)
    for data in data_iter:
        sentences = []
        t = []
        for d in data:
            if d[0] == '.':
                sentences.append(t)
                t = []
            t.append(d[0])
        print('training word 2 vec model')
        model.train(sentences)
        model.save(save_model_file)
    return model