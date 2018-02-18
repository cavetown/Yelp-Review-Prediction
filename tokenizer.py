class tokenizer():
    ''' Create a tokenizer class that will parse the texts until integers
    '''
    def __init__(self, threshold=20):
        self.word2int = {}
        self.threshold = threshold
        self.word_counts = {}

    def _count_words(self, text):
        for sentence in text:
            for word in sentence.split():
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
        print("Size of Vocabulary: " , len(self.word_counts))


    def fit_on_texts(self, texts, embeddings_index):
        ''' Function fits the tokenizer class based on what's available in embeddings index
        :param texts: Type list of lists with with each row containing preprocessed reviews
        :param embeddings_index: Dictionary of words from reviews mapping to an embedding vector from embeddings used
        '''
        self._count_words(texts)

        token_index = 0
        for word, count in self.word_counts.items():
            if count >= self.threshold or word in embeddings_index:
                self.word2int[word] = token_index
                token_index += 1
        special_characters = ["<unk>", "<pad>"]
        for c in special_characters:
            self.word2int[c] = len(self.word2int)

        usage_ratio = round(len(self.word2int) / len(self.word_counts), 4) * 100
        print("Total number of unique words:", len(self.word_counts))
        print("Number of words we will use:", len(self.word2int))
        print("Percent of words we will use: {}%".format(usage_ratio))

    def text_to_sequence(self, text, pred=False):
        '''
        Function to convert a text input into tokenized list.
        :param text: List of lists of reviews to be tokenized
        :param pred: Default is set to false. Tokenizer behaves slightly differently when training versus when making a
                    prediction
        :return: List of lists with each word converted to its tokenized integer
        '''
        if pred:
            seq = []
            for word in text.split():
                if word in self.word2int:
                    seq.append(self.word2int[word])
                else:
                    seq.append(self.word2int["<unk>"])
            return seq
        else:
            seq = []
            for s in text:
                temp_seq = []
                for word in s.split():
                    if word in self.word2int:
                        temp_seq.append(self.word2int[word])
                    else:
                        temp_seq.append(self.word2int["<unk>"])
                seq.append(temp_seq)
            return seq