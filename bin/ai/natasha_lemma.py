from natasha import MorphVocab, Doc

morph_vocab = MorphVocab()

doc = Doc(text)
sent = doc.sents[0]

for token in doc.tokens:
    token.lemmatize(morph_vocab)
