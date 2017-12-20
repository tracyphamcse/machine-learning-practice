import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn import preprocessing
import numpy as np

class Display:
    def __init__(self, word2int, int2word):
        self.word2int = word2int
        self.int2word = int2word

    def euclidean_dist(self, vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))

    def find_top_closest(self, word, embeddings, n_top):

        word_index = self.word2int[word]

        min_dist = 10000 # to act like positive infinity
        min_index = -1

        all_dist = []

        query_vector = embeddings[word_index]

        for index, vector in enumerate(embeddings):

            if not np.array_equal(vector, query_vector):
                all_dist.append({'index' : self.int2word[index], 'dist' : self.euclidean_dist(vector, query_vector)})

        newlist = sorted(all_dist, key=lambda k: k['dist'])
        return newlist[:n_top]

    def draw_word_embedding(self, words, embeddings):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        vectors = model.fit_transform(embeddings)

        normalizer = preprocessing.Normalizer()
        vectors =  normalizer.fit_transform(vectors, 'l2')

        fig, ax = plt.subplots(figsize=(20, 10))

        plt.axis([-2, 2, -2, 2])

        for word in words:
            print(word, vectors[word2int[word]][1])
            ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

        plt.show()
