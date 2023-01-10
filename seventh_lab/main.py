import pandas
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=1000, n_features=2, centers=[(20, 20), (40, 40)], cluster_std=[2.2, 2.0])
# # print(X.shape)
# print('X', X)
# print('Y', y)
#
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], s=10, cmap='rainbow', c=y)
# plt.title('Data')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()
#
# kmeans = KMeans(n_clusters=5, init='k-means++')
# y_pred = kmeans.fit_predict(X)
#
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)


############################################
# дорожные камеры
############################################

# записываем в переменную наши данные
excel_data = pandas.read_excel('cams_20221226.xlsx')

# преобразуем в массив numPy данные о широте и долготе
X_new = pandas.DataFrame(excel_data, columns=['Широта', 'Долгота']).to_numpy()

print(X_new)

# процесс кластеризации
kmeans = KMeans(n_clusters=7, init='k-means++')
pred = kmeans.fit_predict(X_new)

# рисуем МКАД
plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], s=1, cmap='rainbow', c=kmeans.labels_)
plt.title('Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(pred)
