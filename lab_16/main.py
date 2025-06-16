from dependencies import check_and_install_libraries
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

check_and_install_libraries()

# Загрузка и предобработка данных
df = pd.read_csv("news_dataset.csv")
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

# Векторизация
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Обучение модели с балансировкой
model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Оптимальный порог через Precision-Recall curve
y_probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[f1_scores.argmax()]

# Предсказание с оптимальным порогом
y_pred = (y_probs >= optimal_threshold).astype(int)

# Отчёт и визуализация
print("Optimal threshold:", optimal_threshold)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Real', 'Predicted Fake'],
            yticklabels=['Actual Real', 'Actual Fake'])
plt.title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
plt.show()

