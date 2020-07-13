import nltk
import pdfplumber

file = pdfplumber.open('test.pdf')
text = ""

for page in file.pages:
    text += page.extract_text()


###NLTK
#tokenizer = nltk.tokenize.TreebankWordTokenizer()
#print(tokenizer.tokenize(text))

print(text)
tokenize = nltk.tokenize.sent_tokenize(text)
print(tokenize)
