from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

iris=load_iris()

X=iris.data
y=iris.target

scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)

pca=PCA(n_components=2)

X_pca=pca.fit_transform(X_scaled)
print(X_scaled)
print()
print(X_pca)

plt.figure()

for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label = iris.target_names[i])


plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA ile Boyut İndirgeme")
plt.legend()
plt.show()

tsne=TSNE(n_components=2,random_state=42)

X_tsne=tsne.fit_transform(X_scaled)

plt.figure()

for i in range(len(iris.target_names)):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label = iris.target_names[i])

plt.xlabel("X_tsne 1")
plt.ylabel("X_tsne 2")
plt.title("X_tsne ile Boyut İndirgeme")
plt.legend()
plt.show()