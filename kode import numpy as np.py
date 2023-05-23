import numpy as np
from sklearn.mixture import GaussianMixture

# Membuat dataset untuk training dan testing
train_data = np.random.rand(10, 3)
test_data = np.random.rand(1, 3)

# Membuat model Gaussian Mixture Model
model = GaussianMixture(n_components=2)

# Melatih model dengan dataset training
model.fit(train_data)

# Memprediksi label dari data testing
predicted_label = model.predict(test_data)
print(predicted_label)