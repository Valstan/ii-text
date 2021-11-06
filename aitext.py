from clean_text import clean_text
from get_posts import get_posts

from tokenizer_words import tokenizer_words

posts = get_posts(30)

text = []
for i in posts:
    if 'text' in i:
        if i['text']:
            text = clean_text(i['text'])
            text_sequences = tokenizer_words(text)
            print(text_sequences)
