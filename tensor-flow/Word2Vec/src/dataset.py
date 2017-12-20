import numpy as np
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize

class Dataset:
    def __init__(self, language, path, window_size = 1, batch_size = 128):
        self.language = language
        self.path = path

        self.window_size = window_size
        self.batch_size = batch_size

        self.listfile = []
        self.raw_data = []

        self.vocab_size = 0
        self.words = []
        self.word2int = {}
        self.int2word = {}
        self.training_data = []
        self.x_training_batches = []
        self.y_training_batches = []
        self.num_of_batches = 0

    def write_info(self):
        print self.vocab_size
        print self.num_of_batches

    def read_file(self, filename):
        try:
            with open(filename) as f:
                print filename
                return f.read().decode('utf-8', 'ignore').lower()
        except Exception, e:
            print filename, e
            return ""

    def read_files(self):

        print "Reading text files...."

        self.listfile = [f for f in listdir(self.path) if isfile(join(self.path, f))]

        for filename in self.listfile:
            self.raw_data.append(self.read_file(self.path + filename))

    def preprocess_sentence(self, data_raw, per_of_remove):

        '''
            Clean up data and create training pairs:
                - Tokenize word, ignore the punctuation
                - Training data is the pair of words which appear in WINDOW_SIZE in sentence

            Return :
                - sentences: cleaned-up sentences
                - data : list of word pairs [word_0, word_1]
        '''

        corpus_raw = ""
        for data in data_raw:
            corpus_raw = corpus_raw + " " + data

        raw_sentences = corpus_raw.split('.')
        number_of_sentences = len(raw_sentences)

        from_n = int(number_of_sentences*per_of_remove)
        to_n = number_of_sentences - from_n

        sentences = []
        for sentence in raw_sentences[from_n : to_n]:
            sentences.append(word_tokenize(sentence, language=self.language))

        data = []
        for sentence in sentences:
            for word_index, word in enumerate(sentence):
                for nb_word in sentence[max(word_index - self.window_size, 0) : min(word_index + self.window_size, len(sentence)) + 1] :
                    if nb_word != word:
                        data.append([word, nb_word])

        return sentences, data

    def preprocess_word(self, sentences):

        '''
            Create vocab by mapping word2int and int2word

            Return :
                - vocab_size : int
                - word2int : dict (word, int)
                - int2word : dict (int, word)
        '''

        words = []
        for sentence in sentences:
            words.extend(sentence)

        words = set(words)

        word2int = {}
        int2word = {}

        vocab_size = len(words) # gives the total number of unique words

        for i,word in enumerate(words):
            word2int[word] = i
            int2word[i] = word

        return vocab_size, words, word2int, int2word

    def get_mini_batches(self, x_train, y_train, batch_size):
        if batch_size > len(x_train):
            return [x_train], [y_train]

        x_train_batches = []
        y_train_batches = []


        for i in range(0, len(x_train) - batch_size, batch_size):

            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

            for j in range (i, i + batch_size):
                batch[j-i] = x_train[j]
                context[j-i] = y_train[j]

            x_train_batches.append(batch)
            y_train_batches.append(context)

        return x_train_batches, y_train_batches

    def preprocess(self, per_of_remove = 0):

        print "Preprocess sentences..."
        sentences, self.training_data = self.preprocess_sentence(self.raw_data, per_of_remove)

        print "Preprocess words..."
        self.vocab_size, self.words, self.word2int, self.int2word = self.preprocess_word(sentences)

        x_train = []
        y_train = []

        for data_word in self.training_data:
            x_train.append(self.word2int[data_word[0]])
            y_train.append(self.word2int[data_word[1]])

        print "Create training batches..."
        self.x_training_batches, self.y_training_batches = self.get_mini_batches(x_train, y_train, self.batch_size)

        print "DONE"
