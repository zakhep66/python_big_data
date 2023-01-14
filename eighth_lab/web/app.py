from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def hello_world():
    import pandas
    from sklearn.cluster import KMeans

    # записываем в переменную наши данные
    excel_data = pandas.read_csv('colors_1.csv')

    # преобразуем в массив numPy данные о цвете
    X = pandas.DataFrame(excel_data, columns=['r', 'g', 'b']).to_numpy()

    # процесс кластеризации
    kmeans = KMeans(n_clusters=8, init='k-means++')
    pred = kmeans.fit_predict(X)

    data_color = pandas.DataFrame(excel_data, columns=['r', 'g', 'b']).values.tolist()
    res = []

    # все кластеры друг за другом
    for i in range(len(kmeans.cluster_centers_)):
        pred_count = 0
        print(i, 'мега')
        for row in data_color:
            if pred[pred_count] == i:
                res.append([row, pred[pred_count]])
            pred_count += 1

    # все цвета по порядку изначальной последовательности
    # pred_count = 0
    # for i in data_color:
    #     res.append([i, pred[pred_count]])
    #     pred_count += 1

    excel_data_mono = []
    for index, row in excel_data.iterrows():
        mono = int(0.299 * row["r"] + 0.587 * row["g"] + 0.114 * row["b"])
        excel_data_mono.append([mono, mono, mono])

    # классификатор
    model_mono = KMeans(n_clusters=3, init='k-means++')
    prediction_mono = model_mono.fit_predict(excel_data_mono)  # предположение о номере кластера
    cluster_centers_mono = model_mono.cluster_centers_  # список значений центроидов кластеров

    res_mono = []

    # все кластеры друг за другом
    for i in range(len(model_mono.cluster_centers_)):
        pred_count = 0
        print(i, 'мега')
        for row in excel_data_mono:
            if prediction_mono[pred_count] == i:
                res_mono.append([row, prediction_mono[pred_count]])
            pred_count += 1

    # все цвета по порядку изначальной последовательности
    # pred_count = 0
    # for i in excel_data_mono:
    #     res_mono.append([i, prediction_mono[pred_count]])
    #     pred_count += 1

    return render_template('index.html', res=res, cl_center=kmeans.cluster_centers_, res_mono=res_mono,
                           cl_center_mono=cluster_centers_mono)


if __name__ == '__main__':
    app.run(debug=True)
