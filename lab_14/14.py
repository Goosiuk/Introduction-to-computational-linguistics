import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



# 1. Загрузка текста
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


file_path = 'text.txt'
text = load_text(file_path)

# 2. Токенизация
tokens_split = text.split()
tokens_regex = re.findall(r'\b\w+\b', text)
tokens_nltk = word_tokenize(text, language='russian')

print("Токены (.split()):", tokens_split[:10])
print("Токены (regex):", tokens_regex[:10])
print("Токены (NLTK):", tokens_nltk[:10])

# 3. Подсчёт слов
word_count_len = len(tokens_nltk)
word_count_counter = Counter(tokens_nltk)
total_words_counter = sum(word_count_counter.values())

print("Количество слов (len):", word_count_len)
print("Количество слов (Counter):", total_words_counter)

# 4. Очистка
tokens_cleaned = [token for token in tokens_nltk if token.isalpha()]
stop_words = set(stopwords.words('russian'))
tokens_no_stopwords = [token for token in tokens_cleaned if token.lower() not in stop_words]

print("Токены без пунктуации:", tokens_cleaned[:10])
print("Токены без стоп-слов:", tokens_no_stopwords[:10])

# 5. Стемминг
stemmer = SnowballStemmer('russian')
tokens_stemmed = [stemmer.stem(token) for token in tokens_no_stopwords]
print("Стеммированные токены:", tokens_stemmed[:10])
