import nltk
import os
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
import json
from bson import json_util

st = StanfordNERTagger('/usr/share/stanford/stanford-ner-2016-10-31/classifiers/english.muc.7class.distsim.crf.ser.gz',
		'/usr/share/stanford/stanford-ner-2016-10-31/stanford-ner.jar')
class NamedEntityRecognition:

	def tokenize_sentences(self,sentences):
		tokenized_words = [nltk.word_tokenize(sentence) for sentence in sentences]
		return tokenized_words

	def bio_tagger(self,tagged_sentences,output):
		named_entities = {'ORGANIZATION':[],'LOCATION':[],'PERSON':[],'MONEY':[],'PERCENT':[],'DATE':[],'TIME':[]}
		named_entities1 = []
		for tagged_sentence in tagged_sentences:
			prev_tag = 'O'
			NE = ""
			for token,tag in tagged_sentence:
				if tag=="O":
					if NE!='' and prev_tag!='O':
						named_entities1.append((NE,prev_tag))
						named_entities[prev_tag].append(NE)
					prev_tag = "O"
					NE=""
					continue
				if tag != "O" and prev_tag == "O":
					NE = token
					prev_tag = tag
				elif prev_tag != "O" and prev_tag == tag:
					NE = NE+" "+token
					prev_tag = tag
				elif prev_tag != "O" and prev_tag != tag:
					named_entities1.append((NE,prev_tag))
					named_entities[prev_tag].append(NE)
					prev_tag = tag
					NE = token
			if prev_tag!='O':
				named_entities1.append((NE,prev_tag))
				named_entities[prev_tag].append(NE)
		for key in named_entities.keys():
			named_entities[key] = list(set(named_entities[key]))
		if output: return list(set(named_entities1)) 
		else: return named_entities

	def ne_recognizer(self,data,output=True):
		sentences = nltk.sent_tokenize(data)
		tokenized_sentences = self.tokenize_sentences(sentences)
		tagged_sentences = st.tag_sents(tokenized_sentences)
		bio_tagged = self.bio_tagger(tagged_sentences,output)
		return bio_tagged
