import nltk
import pdfplumber

file = pdfplumber.open('test.pdf')
text = ""

for page in file.pages:
    text += page.extract_text()

#tokenizer = nltk.tokenize.TreebankWordTokenizer()
#print(tokenizer.tokenize(text))
print(text)
