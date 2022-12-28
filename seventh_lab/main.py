import pandas
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, n_features=2, centers=[(20, 20), (40, 40)], cluster_std=[2.2, 2.0])
# print(X.shape)
print('Хуй', X)
print('хУй', y)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=10, cmap='rainbow', c=y)
plt.title('Data')
plt.xlabel('x1')
plt.ylabel('x1')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++')
y_pred = kmeans.fit_predict(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)


############################################
# дорожные камеры
############################################

# excel_data = pandas.read_excel('cams_20221226.xlsx')
#
# X_new = pandas.DataFrame(excel_data, columns=['Широта', 'Долгота']).to_numpy()
#
# print(X_new)
#
# plt.figure()
# plt.scatter(X_new[:, 0], X_new[:, 1], s=10, cmap='rainbow', c=y_new)
# plt.title('Data')
# plt.xlabel('x1')
# plt.ylabel('x1')
# plt.show()
