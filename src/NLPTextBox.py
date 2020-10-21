from pdfminer.layout import LTTextBox
from pdfminer.layout import LTTextLine
from pdfminer.layout import LTChar

import nltk
#nltk.download('all')

def sum_font(item):
    b = 0
    i = 0
    size = 0
    length = 0
    for child in item:
        if isinstance(child, LTChar):
            size += child.size
            length += 1
            if 'Bold' in child.fontname:
                b += 1
            elif 'Italic' in child.fontname:
                i += 1
        elif isinstance(child, LTTextLine):
            _b, _i, _size, _length = sum_font(child)
            b += _b
            i += _i
            size += _size
            length += _length
        elif isinstance(child, LTTextBox):
            _b, _i, _size, _length = sum_font(child)
            b += _b
            i += _i
            size += _size
            length += _length
    return (b, i, size, length)

class NLPTextBox(LTTextBox):
    def __init__(self, box):
        LTTextBox.__init__(self)
        self.bbox = box.bbox
        self.index = box.index
        self.b, self.i, self.size, self.length = sum_font(box)
        self.text = box.get_text()
        self.tag = ""
        self.key = ""
        self.list_tag = False

    def set_tag(self, tag):
        if (tag == "paragraph"):
            tokens = nltk.word_tokenize(self.text)
            self.key = tokens[0]
            self.text = self.text.replace(self.key, '')
        elif (tag == "section"):
            tokens = nltk.word_tokenize(self.text)
            self.key = tokens[0]
            self.text = self.text.replace(self.key, '')
        elif (tag == "subsection"):
            tokens = nltk.word_tokenize(self.text)
            self.key = tokens[0]
            self.text = self.text.replace(self.key, '')
        elif (tag == "li"):
            tokens = nltk.word_tokenize(self.text)
            if (")" in tokens[1]):
                self.key = tokens[0]
                self.text = self.text.replace(tokens[0] + tokens[1], '')
                self.list_tag = True
            else:
                self.key = self.text
        else:
            self.key = self.text

        self.tag = tag
    
    def get_text(self):
        t = ' '.join(self.text.replace('\n', ' ').lstrip().rstrip().split())
        return t

    def get_key(self):
        return self.key

class NLPSimpleBox():
    def __init__(self, tag, key):
        self.text = ""
        self.tag = tag
        self.key = key
    
    def get_text(self):
        t = ' '.join(self.text.replace('\n', ' ').lstrip().rstrip().split())
        return t

    def get_key(self):
        return self.key