from spacy.lang.en import English

# Section 1: Understanding tokens
nlp = English()
doc = nlp("Let us get started with demos, begining with understanding tokens!")

for token in doc:
    print(token)

print('\n')

''
# Section 2: Understanding attributes of a token in 'doc' object
nlp = English()
doc = nlp("Ten years ago, Mumbai Metro project became operational on June 8th")

fmt_str = "{:<12}| {:<10}| {:<10}| {:<10}"
print(fmt_str.format("token", "is_alpha","is_punct", "like_num"))

for token in doc:
    print(fmt_str.format(token.text, token.is_alpha, token.is_punct, token.like_num))

print('\n')
''

# Section 3: Understanding trained pipelines: stats models to predict tokens
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Ten years ago, Mumbai Metro project became operational on June 8th")
#doc = nlp("Ten years ago, Mumbai metro project became operational on June 8th")

fmt_str = "{:<12}| {:<6}| {:<8}| {:<8}"
print(fmt_str.format("token", "pos", "label", "parent"))

# pos - Parts Of Speech tag, ent_type - Named Entity Recognition
for token in doc:
    print(fmt_str.format(token.text, token.pos_, token.ent_type_, token.head.text))

''

# Homework: Go through video tuts on Spacy to master advanced NLP concepts :)
# Did you know -- Spacy enables industrial strength NLP in python!
#
# https://spacy.io/ [Spacy Homepage]
# https://course.spacy.io/en/ [FREE and interactive online course includes 55 exercises
#           featuring videos, slide-decks, MCQs, and interactive coding in browser!]