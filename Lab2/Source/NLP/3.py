import nltk

# Read the data from a file
infile = open('C:/Users/Dave/Documents/GitHub/CSS-5590-001-Python-Deep-Learning/Lab2/Source/NLP/nlp_input.txt')
data = infile.read()

# Tokenize the text into words and apply lemmatization technique on each word
lemma_list = list()
words = nltk.word_tokenize(data)
lemmatizer = nltk.stem.WordNetLemmatizer()
for word in words:
    lemma = lemmatizer.lemmatize(word)
    lemma_list.append(lemma)
print(lemma_list, '\n')

# Find all the trigrams for the words
trigrams = nltk.trigrams(words)
for trigram in trigrams:
    print(trigram)

# Extract the top 10 of the most repeated trigrams based on their count
trigrams = nltk.trigrams(words)
trigram_dict = dict()
for trigram in trigrams:
    if trigram not in trigram_dict.keys():
        trigram_dict[trigram] = 1
    else:
        trigram_dict[trigram] += 1

top_10_list = list()
for i in range(0,10):
    max_count = 0
    max_trigram = 0
    for trigram in trigram_dict.keys():
        if trigram_dict[trigram] > max_count:
            max_count = trigram_dict[trigram]
            max_trigram = trigram
            trigram_dict[trigram] = 0
    top_10_list.append(max_trigram)

print('\n', top_10_list, '\n')

# Go through the text in the file
# Find all the sentences with the most repeated tri-grams
# Extract those sentences and concatenate

concatenated = ''
sentences = nltk.sent_tokenize(data)
for sentence in sentences:
    sentence_words = nltk.word_tokenize(sentence)
    sentence_trigrams = nltk.trigrams(sentence_words)
    found = False
    for trigram in sentence_trigrams:
        if trigram in top_10_list:
            found = True
    if found:
        concatenated += sentence

# Print the concatenated result
print(concatenated)
