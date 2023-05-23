from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Membuat dataset untuk training dan testing
train_data = ['Ini komentar positif', 'Ini komentar negatif', 'Ini komentar netral']
train_labels = ['positif', 'negatif', 'netral']
test_data = ['Ini komentar baru']

# Mengubah teks menjadi vektor dengan metode TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Membuat model Naive Bayes
clf = MultinomialNB()

# Melatih model dengan dataset training
clf.fit(train_vectors, train_labels)

# Memprediksi label dari komentar baru
predicted_label = clf.predict(test_vectors)
print(predicted_label)