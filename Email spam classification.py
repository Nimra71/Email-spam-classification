# import pandas as pd
# import numpy as np

# # Load CSV safely on Windows
# df = pd.read_csv(r"C:\Users\LAPTOPS HUB\Downloads\archive\spam.csv", encoding='latin-1')

# # Keep only relevant columns
# df = df[['v1', 'v2']]
# df.columns = ['label', 'text']

# # Convert label to numeric
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# # Check data
# print(df.head())
# print("Dataset size:", df.shape)


# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess(text):
#     text = text.lower()  # convert to lowercase
#     text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
#     words = text.split()  # split sentence into words
#     words = [stemmer.stem(word) for word in words if word not in stop_words]
#     return " ".join(words)

# df['clean_text'] = df['text'].apply(preprocess)
# df.head()

# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
# X = vectorizer.fit_transform(df['clean_text']).toarray()
# y = df['label']

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\LAPTOPS HUB\Downloads\archive\spam.csv", encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham':0,'spam':1})

sns.countplot(x=df['label'])
plt.title("Spam vs Ham Distribution")
plt.xticks([0,1], ["Ham","Spam"])
plt.show()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess)

from collections import Counter

spam_words = " ".join(df[df['label']==1]['clean_text']).split()
ham_words = " ".join(df[df['label']==0]['clean_text']).split()

spam_common = Counter(spam_words).most_common(20)
ham_common = Counter(ham_words).most_common(20)

spam_df = pd.DataFrame(spam_common, columns=['word','count'])
ham_df = pd.DataFrame(ham_common, columns=['word','count'])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='count', y='word', data=spam_df)
plt.title("Top Words in Spam")

plt.subplot(1,2,2)
sns.barplot(x='count', y='word', data=ham_df)
plt.title("Top Words in Ham")

plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_probs = model.feature_log_prob_[1]

top_spam_words = sorted(zip(spam_probs, feature_names), reverse=True)[:20]
top_words_df = pd.DataFrame(top_spam_words, columns=["Score", "Word"])

sns.barplot(x="Score", y="Word", data=top_words_df)
plt.title("Top Words Indicating Spam")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear SVM": LinearSVC()
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, preds))
