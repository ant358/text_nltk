import logging
import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class Keywords():
    """
    Class to store keywords and their frequency
     """

    def __init__(self, text) -> None:
        self.text = text
        self.lemmitiser = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)
        self.tokens = self.tokenize()
        self.no_punc = self.strip_punctuation()
        self.no_ents = self.remove_suspected_entities()
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.no_stopwords = self.remove_stopwords()
        self.tagged = self.tagger()
        self.nouns = self.extract_nouns()
        self.lemit_nouns = [self.lemmitiser.lemmatize(w) for w in self.nouns]
        self.top_nouns = self.return_top_keywords(self.lemit_nouns, 20)
        self.verbs = self.extract_verbs()
        self.lemit_verbs = [self.lemmitiser.lemmatize(w) for w in self.verbs]
        self.top_verbs = self.return_top_keywords(self.lemit_verbs, 20)
        self.top_words = self.return_top_keywords(self.no_stopwords, 20)

    def tokenize(self):
        try:
            return nltk.word_tokenize(self.text)
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            return []

    def strip_punctuation(self):
        try:
            return [w for w in self.tokens if w.isalpha()]
        except Exception as e:
            self.logger.error(f"Error stripping punctuation: {e}")
            return []

    def remove_suspected_entities(self):
        try:
            return [w for w in self.no_punc if not w[0].isupper()]
        except Exception as e:
            self.logger.error(f"Error removing suspected entities: {e}")
            return []

    def remove_stopwords(self):
        try:
            return [w for w in self.no_ents if w not in self.stopwords]
        except Exception as e:
            self.logger.error(f"Error removing stopwords: {e}")
            return []

    def tagger(self):
        try:
            return nltk.pos_tag(self.no_stopwords)
        except Exception as e:
            self.logger.error(f"Error tagging text: {e}")
            return []

    def extract_nouns(self):
        try:
            return [
                w for w, pos in self.tagged
                if pos in ['NN', 'NNP', 'NNS', 'NNPS']
            ]
        except Exception as e:
            self.logger.error(f"Error extracting nouns: {e}")
            return []

    def extract_verbs(self):
        try:
            return [
                w for w, pos in self.tagged
                if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            ]
        except Exception as e:
            self.logger.error(f"Error extracting verbs: {e}")
            return []

    def return_top_keywords(self, word_list, list_size):
        try:
            fdist = FreqDist(word_list)
            return dict(fdist.most_common(list_size))
        except Exception as e:
            self.logger.error(f"Error extracting top keywords: {e}")
            return {}
