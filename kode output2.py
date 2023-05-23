from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Memuat dataset
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Mengubah teks menjadi vektor dengan metode TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data.data)
test_vectors = vectorizer.transform(test_data.data)

# Membuat model Naive Bayes
clf = MultinomialNB()

# Melatih model dengan dataset training
clf.fit(train_vectors, train_data.target)

# Memprediksi label dari dataset testing
predicted_labels = clf.predict(test_vectors)

# Mengukur akurasi
accuracy = accuracy_score(test_data.target, predicted_labels)
print("Akurasi:", accuracy)
