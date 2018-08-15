import os
import pickle
import re
import random

import nltk.data
import pymorphy2

from nltk import word_tokenize, ToktokTokenizer, RegexpTokenizer
from stop_words import get_stop_words
from tqdm import tqdm



class Document:
    def __init__(self, title: str, text: str):
        self.title = title
        self.main_text = _parse_paragraphs(text)

    def __repr__(self):
        s = '{}\n{}'.format(self.title, self.get_text())
        return s


    def get_text(self):
        return '\n'.join([paragraph.get_text() for paragraph in self.main_text])

  
class Paragraph:  
    def __init__(self, text: str = ''):
        self.sentences = _parse_sentences(text)

    def get_sentences(self, normalize=False, tokenize=False, remove_stop_words=False):
        return self.sentences

    def get_text(self):        
        return '\n'.join([sentence.get_text() for sentence in self.sentences])
    
    def __repr__(self):
        return ' '.join([sentence.get_text() for sentence in self.sentences])


class Sentence:
    def __init__(self, sentence: str):
        self.sentence = sentence
        
    def tokenize(self):
        return word_tokenize(self.sentence)

    def regexp_tokenize(self):
        tokenizer = RegexpTokenizer(r'[а-яА-ЯA-Za-z]+')
        return tokenizer.tokenize(self.sentence)

    def remove_stop_words(self):
        pattern = re.compile(r'[^\w-]')
        stop_words = get_stop_words('russian')
        tokens = self.tokenize()
        good_tokens = []
        for token in tokens:
            if not pattern.search(token) and token.lower() not in stop_words:
                good_tokens.append(token)
        return Sentence(' '.join(good_tokens))    
    
    def get_text(self):
        return self.sentence

    def normalize(self):
        morph = pymorphy2.MorphAnalyzer()
        tokens = self.tokenize()
        norm_tokens = []
        for token in tokens:
            norm_tokens.append(morph.parse(token)[0].normal_form)
        return Sentence(' '.join(norm_tokens))

    def __repr__(self):
        return self.sentence

    
def _parse_sentences(text: str):
    text = re.sub(r'[\t\n\r ]', ' ', text)    

    tokenizer = nltk.data.load('russian.pickle')
    special_abbr = ['млн', 'коп', 'руб', 'ук']
    for abbr in special_abbr:
        tokenizer._params.abbrev_types.add(abbr)

    sentences = list(map(lambda x: Sentence(x.strip()), tokenizer.tokenize(text)))
    return sentences


def _parse_paragraphs(doc_text: str):
    lines = map(lambda x: x.strip(), doc_text.split('\n'))    
    return [Paragraph(line) for line in lines if line]


def _parse_file(filename: str):    
    with open(filename, 'r', encoding='utf-8') as f:
        def next_line():        
            l = f.readline()
            return None if len(l) == 0 else l.strip()
                
        line = next_line()
        while (line is not None):
            while (line is not None) and line != 'Название документа':
                line = next_line()
            line = next_line()

            title = ''
            while (line is not None) and line != 'Текст документа':
                title += line + '\n'
                line = next_line()
            line = next_line()
            
            text = ''
            while (line is not None) and not line.startswith('-----'):
                text += line + '\n'
                line = next_line() 

            yield Document(title, text)

    
    