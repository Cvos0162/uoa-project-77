import nltk
text = "Sentence Test 1! Sentence Test 2. Sentece Test 3? Sentence Test 4"

tokenize = nltk.tokenize.sent_tokenize(text)
print(tokenize)
