import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import nltk
import re
#step 2
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#step 3
data = {
    'text': [
        "I am so happy today!",
        "This is frustrating and annoying.",
        "I feel very sad and alone.",
        "Wow, that was unexpected!",
        "I’m scared of what's going to happen.",
        "I love this!",
        "Why did this happen to me?",
        "This is the best day ever!",
        "I can't stop crying.",
        "This makes me furious!"
    ],
    'emotion': [
        'joy', 'anger', 'sadness', 'surprise', 'fear',
        'joy', 'sadness', 'joy', 'sadness', 'anger'
    ]
}

df = pd.DataFrame(data)
#step 4
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['text'].apply(clean_text)
#step 5
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['emotion']
#step 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#step 7
model = LogisticRegression()
model.fit(X_train, y_train)
#step 8
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Emotion Classification Confusion Matrix")
plt.show()
