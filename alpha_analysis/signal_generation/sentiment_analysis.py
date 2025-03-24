import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Загрузка необходимых ресурсов NLTK
nltk.download('vader_lexicon')


class SentimentAnalysis:
    def __init__(self, texts):
        """
        A class for analyzing the tone of texts.

        :param texts: List[str] or pd.Series - an array of texts (e.g., news headlines).
        """
        self.texts = texts
        self.sia = SentimentIntensityAnalyzer()

    def vader_sentiment(self):
        """
        Tonality analysis with VADER.

        :return: DataFrame with Sentiment metrics (compound, pos, neu, neg).
        """
        sentiments = [self.sia.polarity_scores(text) for text in self.texts]
        df = pd.DataFrame(sentiments)
        df.insert(0, 'Text', self.texts)
        return df

    def textblob_sentiment(self):
        """
        Analyzing tone with TextBlob.

        :return: DataFrame with Sentiment metrics (polarity and subjectivity).
        """
        sentiments = [{'Polarity': TextBlob(text).sentiment.polarity,
                       'Subjectivity': TextBlob(text).sentiment.subjectivity} for text in self.texts]
        df = pd.DataFrame(sentiments)
        df.insert(0, 'Text', self.texts)
        return df

    def combined_sentiment(self):
        """
        Merged analysis using VADER and TextBlob.

        :return: DataFrame with merged metrics.
        """
        vader_df = self.vader_sentiment()
        textblob_df = self.textblob_sentiment()
        combined_df = vader_df.merge(textblob_df, on="Text")
        return combined_df
