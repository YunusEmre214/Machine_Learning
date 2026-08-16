import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Veri Setinin Yüklenmesi
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target # 0: Kötü Huylu (Malignant), 1: İyi Huylu (Benign)

print("Veri Seti Boyutu:", X.shape)
print("\nVeri Setinden İlk 5 Satır:\n", X.head())

# 2. Veri Ön İşleme (Train/Test Split & Scaling)
# Veriyi %80 Eğitim, %20 Test olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN ve SVM uzaklık tabanlı algoritmalar olduğu için veriyi ölçeklendirmek (Standardization) KESİNLİKLE gereklidir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Kurulumu ve Eğitimi
# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_y_pred = knn_model.predict(X_test_scaled)

# Support Vector Machines (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_y_pred = svm_model.predict(X_test_scaled)

# 4. Model Değerlendirme
print("\n--- KNN Model Sonuçları ---")
print("Doğruluk Oranı (Accuracy):", accuracy_score(y_test, knn_y_pred))
print(classification_report(y_test, knn_y_pred))

print("\n--- SVM Model Sonuçları ---")
print("Doğruluk Oranı (Accuracy):", accuracy_score(y_test, svm_y_pred))
print(classification_report(y_test, svm_y_pred))

# 5. Grafikler (Karmaşıklık Matrisi - Confusion Matrix)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, knn_y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("KNN Karmaşıklık Matrisi")
axes[0].set_xlabel("Tahmin Edilen")
axes[0].set_ylabel("Gerçek Değer")

sns.heatmap(confusion_matrix(y_test, svm_y_pred), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("SVM Karmaşıklık Matrisi")
axes[1].set_xlabel("Tahmin Edilen")
axes[1].set_ylabel("Gerçek Değer")

plt.tight_layout()
plt.show()