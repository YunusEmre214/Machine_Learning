import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Veri setinin yüklenmesi ve temel bilgilerin incelenmesi
digits = load_digits()
print(digits.DESCR)

# 2. örnek görüntülerin görselleştirilmesi
fig, axes = plt.subplots(nrows=2, ncols=5, figsize = (8,5), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(f"Label: {digits.target[i]}")

plt.tight_layout()
plt.show()

# 3. özellik ve hedef değişkenlerin ayrılması
X = digits.data
y = digits.target

# 4. eğitim ve test veri setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# 5. svm modeli oluşturma
svm = SVC(kernel="linear", random_state=42)

# 6. modelin eğitilmesi
svm.fit(X_train, y_train)

# 7. test verisi üzerinde tahmin yapılması
y_pred = svm.predict(X_test)

# 8. model performansının sınıflandırma raporu ile değerlendirilmesi
cls_report = classification_report(y_test, y_pred)
print(cls_report)