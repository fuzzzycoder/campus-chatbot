from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize,word_tokenize
import json
from bson import json_util

sid = SentimentIntensityAnalyzer()

class SentimentAnalyzer:
	def sentiment_analysis(self, sentence):
		## Sentiment for whole document
		sentiment = []
		if sentence:
			ss = sid.polarity_scores(sentence)
			sentiment.append({sentence:ss})
			return sentiment['compound']
		return 0

