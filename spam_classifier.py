# Email Spam Classifier using NLP + Machine Learning
# Author: Anupam Vishwakarma
# Model: Multinomial Naive Bayes

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download stopwords (only once)

nltk.download('stopwords')

print("===  Email Spam Classifier ===")


# Step 1️ - Load dataset

print("\n Loading dataset...")
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

print(f"Dataset loaded successfully! Total messages: {len(data)}")


# Step 2️ - Preprocess text

print("\n Cleaning text...")

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

data['cleaned_message'] = data['message'].apply(preprocess_text)

print(" Text cleaned successfully!")



# Step 3️ - Feature extraction using TF-IDF

print("\n Converting text to features (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned_message']).toarray()
y = data['label'].map({'ham': 0, 'spam': 1})


# Step 4️ - Split data

print("\n Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5️ - Train model

print("\n Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)


# Step 6️ - Evaluate model

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n Model trained successfully with accuracy: {acc * 100:.2f}%")
print("\n Classification Report:\n", classification_report(y_test, y_pred))


# Step 7️ - Save model and vectorizer

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n Model and vectorizer saved as 'spam_model.pkl' and 'vectorizer.pkl'")


# Step 8️ - Test with custom message

def predict_message(msg):
    msg_clean = preprocess_text(msg)
    msg_vec = vectorizer.transform([msg_clean]).toarray()
    pred = model.predict(msg_vec)[0]
    return " Spam" if pred == 1 else " Not Spam"

print("\n Test the model manually:")
print(predict_message("Congratulations! You’ve won a free iPhone. Click here to claim your prize."))
print(predict_message("Hey Anupam, please find the meeting schedule attached."))

print("\n Done! Your Email Spam Classifier is ready to use.")
