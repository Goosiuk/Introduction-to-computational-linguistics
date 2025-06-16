import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.text import Text
from collections import Counter
import pymorphy2
import matplotlib.pyplot as plt
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

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

# 6. Лемматизация
morph = pymorphy2.MorphAnalyzer()
tokens_lemmatized = [morph.parse(token)[0].normal_form for token in tokens_no_stopwords]
print("Лемматизированные токены:", tokens_lemmatized[:10])

# 7. График частотности
lemmatized_counter = Counter(tokens_lemmatized)
most_common = lemmatized_counter.most_common(10)
words, counts = zip(*most_common)

plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.title('10 наиболее частотных лемматизированных слов')
plt.xlabel('Слова')
plt.ylabel('Частота')
plt.xticks(rotation=45)
plt.show()

# 8. Dispersion Plot
nltk_text = Text(tokens_lemmatized)
nltk_text.dispersion_plot([word for word, _ in most_common[:5]])

# 9. Грамматические характеристики
pos_tags = [morph.parse(token)[0].tag.POS for token in tokens_no_stopwords if morph.parse(token)[0].tag.POS]
print("Части речи (первые 10):", pos_tags[:10])

# 10. График частей речи
pos_counter = Counter(pos_tags)
pos_labels, pos_counts = zip(*pos_counter.items())

plt.figure(figsize=(10, 6))
plt.bar(pos_labels, pos_counts, color='lightgreen')
plt.title('Распределение частей речи')
plt.xlabel('Часть речи')
plt.ylabel('Частота')
plt.show()

# 11. Тестирование методов NLTK
nltk_text = Text(tokens_nltk)
print("Слова, похожие на 'книга':")
nltk_text.similar('книга')

print("\nОбщие контексты для 'книга' и 'автор':")
nltk_text.common_contexts(['книга', 'автор'])

print("\nКоллокации:")
nltk_text.collocations()