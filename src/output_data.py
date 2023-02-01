# %%
import logging
import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)


class Keywords():
    """
    Class to store keywords and their frequency
     """

    def __init__(self, text_json) -> None:
        self.text = text_json['text']
        self.pageId = text_json['pageId']
        self.lemmitiser = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)
        self._tokens = self._tokenize()
        self._no_punc = self._strip_punctuation()
        self._no_ents = self._remove_suspected_entities()
        self._stopwords = nltk.corpus.stopwords.words('english')
        self._no_stopwords = self._remove_stopwords()
        self._tagged = self._tagger()
        self._nouns = self._extract_nouns()
        self._lemit_nouns = [self.lemmitiser.lemmatize(w) for w in self._nouns]
        self.top_nouns = self.return_top_keywords(self._lemit_nouns, "Noun",
                                                  20)
        self._verbs = self._extract_verbs()
        self._lemit_verbs = [self.lemmitiser.lemmatize(w) for w in self._verbs]
        self.top_verbs = self.return_top_keywords(self._lemit_verbs, "Verb",
                                                  20)
        self.top_words = self.return_top_keywords(self._no_stopwords, "All",
                                                  20)

    def _tokenize(self):
        try:
            return nltk.word_tokenize(self.text)
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            return []

    def _strip_punctuation(self) -> list[str]:
        try:
            return [w for w in self._tokens if w.isalpha()]
        except Exception as e:
            self.logger.error(f"Error stripping punctuation: {e}")
            return []

    def _remove_suspected_entities(self) -> list[str]:
        try:
            return [w for w in self._no_punc if not w[0].isupper()]
        except Exception as e:
            self.logger.error(f"Error removing suspected entities: {e}")
            return []

    def _remove_stopwords(self) -> list[str]:
        try:
            return [w for w in self._no_ents if w not in self._stopwords]
        except Exception as e:
            self.logger.error(f"Error removing stopwords: {e}")
            return []

    def _tagger(self):    # -> list[tuple[str, str]]:
        try:
            return nltk.pos_tag(self._no_stopwords)
        except Exception as e:
            self.logger.error(f"Error tagging text: {e}")
            return []

    def _extract_nouns(self) -> list[str]:
        try:
            return [
                w for w, pos in self._tagged
                if pos in ['NN', 'NNP', 'NNS', 'NNPS']
            ]
        except Exception as e:
            self.logger.error(f"Error extracting nouns: {e}")
            return []

    def _extract_verbs(self) -> list[str]:
        try:
            return [
                w for w, pos in self._tagged
                if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            ]
        except Exception as e:
            self.logger.error(f"Error extracting verbs: {e}")
            return []

    def return_top_keywords(self, word_list: list, word_type: str,
                            list_size: int) -> dict:
        try:
            fdist = FreqDist(word_list)
            return {
                "pageId": self.pageId,
                "word_type": word_type,
                "keywords": dict(fdist.most_common(list_size))
            }
        except Exception as e:
            self.logger.error(f"Error extracting top keywords: {e}")
            return {}


def load_to_graph_db(docment: dict[str, str], keyuord_results: dict):
    """
    """
    # create the graph database connection
    with GraphDatabase.driver("bolt://host.docker.internal:7687") as driver:
        graph = driver.session()
        # get the first key value
        if len(keyuord_results['keywords']) > 0:
            query = (
                "MATCH (d:Document {pageId: $document_id}) "
                "MERGE (e:Keyword {word: $word}) "
                "MERGE (d)-[:HAS_KEYWORD {frequency: $freq, word_type: $word_type}]->(e) "
            )
            for k, v, in keyuord_results['keywords'].items():
                try:
                    # create the entity node and the relationship
                    result = graph.run(
                        query,
                        document_id=keyuord_results['pageId'],
                        word_type=keyuord_results['word_type'],
                        word=k,
                        freq=v,
                    )
                    logger.info(
                        f"Created keyword node for {k} in document {keyuord_results['pageId']} - {result}"
                    )
                except ServiceUnavailable as e:
                    logger.exception(e)
                    logger.error(
                        f"During {keyuord_results['pageid']} could not connect to the graph database for keyword creation"
                    )


if __name__ == "__main__":

    text = """The oversized snowflakes fell softly and silently, settling among the pines like a picturesque Christmas scene.
    By the roadside, spectators in heavy winter coats watched team cars and motorbikes struggle up one of Liege-Bastogne-Liege's countless climbs, tyres spinning in the slush as they pursued one man on a bike.
    It was April 1980 and Bernard Hinault, almost unrecognisable beneath a big red balaclava, slewed doggedly on, further into the lead, somehow remaining balanced on the two wheels beneath him.
    He was under such physical strain that he would do himself permanent damage. Pushing his body to its very limit, he raced through the Ardennes in search of victory in the race known as 'La Doyenne' - the old lady.
    So bad were the conditions that several of cycling's best riders collected their number from organisers and then never lined up.
    After just 70km of the 244km one-day race, 110 of the 174 entrants were already holed up in a hotel by the finish line. Only 21 completed the course. Hinault suffered frostbite.
    Rarely do you see such attrition in cycling, but Liege-Bastogne-Liege, which celebrates its 130th birthday on Sunday, has been making and breaking the toughest competitors for years.
    Hinault was 25. He had already won the Tour de France twice and would go on to win it a further three times, an icon of his sport in the making. His total of five Tour victories remains a joint record.
    But this was a different challenge - a long way from the searing heat and sunflowers of summer.
    One of the five prestigious 'Monument' one-day races in cycling, Liege-Bastogne-Liege is celebrated by many for being the very antithesis of the Tour.
    In the hills of east and south Belgium the peloton is stretched through thick, damp forest, over short, sharp climbs and across tricky, part-cobbled sections before landing back where it all began in Liege.
    "[The race is] already hard, it's long, and when I won it was in very tough conditions, especially the snow," says Hinault, now aged 67.
    "Yes, I considered quitting if the weather conditions persisted. We started having difficulties. It's difficult in Liege-Bastogne-Liege."
    Hinault's account of one of his greatest triumphs is characteristically taciturn. Tough conditions is a severe understatement. And in the racing he didn't have it all his own way, either.
    With around 91km to go, approaching the 500m Stockeu climb, Rudy Pevenage was two minutes 15 seconds ahead of Hinault and a small chasing group.
    Pevenage was one of the hard men of the spring classics. He was a Belgian with a big lead, in conditions many locals would feel only a Belgian could master.
    But even he did not finish a race that truly separated the men from the legends. 'Neige-Bastogne-Neige,' as it would be dubbed.
    On the next climb, a 500m ascent of the Haute Levee, Hinault and a small number of fellow pursuers caught up with Pevenage. Then Hinault launched his attack, bright red balaclava and thick blue gloves disappearing into the distance as his stunning acceleration left everybody behind.
    There were still 80km to go.
    """
    # test the keyword extraction
    keyword_results = Keywords({"pageId": "1234", "text": text})
    print(keyword_results.top_nouns)

# %%
