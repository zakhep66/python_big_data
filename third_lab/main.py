import pandas
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('../second_lab/normalized_data_set.csv')

########################################################################
# Выбросы
########################################################################

columns = [
    "gender_male", "gender_female", "age", "height_bad", "timestamp", "dead"
]

for column in columns:
    out = df.loc[
        (df[column] < (df[column].mean() - 3 * df[column].std())) | (
                df[column] > (
                df[column].mean() + 3 * df[column].std())), column]
    print(out)
    out_indexes = df.index[
        (df[column] < (df[column].mean() - 3 * df[column].std())) | (
                df[column] > (
                df[column].mean() + 3 * df[column].std()))].tolist()
    print(out_indexes)
    df.drop(df.index[out_indexes], axis=0, inplace=True)
print(df)

########################################################################
# Тестовые и тренировочные данные
########################################################################


X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1],
    df['dead'],
    test_size=0.25,
    random_state=0
)

print(X_train.shape, X_test.shape)

########################################################################
# Создание и обучение классификатора
########################################################################

# LabelEncoder
lab = preprocessing.LabelEncoder()

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

knn.fit(X_train, y_train)

########################################################################
# Прогнозирование значений меток для тестового набора
########################################################################

prediction = knn.predict(X_test)
print(prediction)
print(X_test.assign(predict=prediction))

########################################################################
# Оценка качества работы классификатора
########################################################################

print(format(knn.score(X_test, y_test)))

########################################################################
# Третий пункт задания
########################################################################

# данные для проверки
some_norm_data_set2 = {
    'gender_male': 1.0,
    'gender_female': 0.0,
    'age': 0.45,
    'height_bad': 11,
    'timestamp': 0.996
}
some_norm_data_set = {
    'gender_male': 0.0,
    'gender_female': 1.0,
    'age': 0.93,
    'height_bad': 1.0,
    'timestamp': 0.996
}
some_norm_data_set3 = {
    'gender_male': 1.0,
    'gender_female': 0.0,
    'age': 0.98,
    'height_bad': 0.28571428571428575,
    'timestamp': 0.996
}


# проверяем только что сгенеренный dataset на выбросы и помечаем норм/не норм данные
min_v = [
    df['gender_male'].mean() - 3 * df['gender_male'].std(),
    df['gender_female'].mean() - 3 * df['gender_female'].std(),
    df['age'].mean() - 3 * df['age'].std(),
    df['height_bad'].mean() - 3 * df['height_bad'].std(),
    df['timestamp'].mean() - 3 * df['timestamp'].std()
]  # минус три сигма

max_v = [
    df['gender_male'].mean() + 3 * df['gender_male'].std(),
    df['gender_female'].mean() + 3 * df['gender_female'].std(),
    df['age'].mean() + 3 * df['age'].std(),
    df['height_bad'].mean() + 3 * df['height_bad'].std(),
    df['timestamp'].mean() + 3 * df['timestamp'].std()
]  # плюс три сигма

print(max_v, min_v, sep='\n')

some_columns = [
    'gender_male', 'gender_female', 'age', 'height_bad', 'timestamp'
]  # столбцы по которым проводим проверку на выбросы

is_good_data = False

for column in some_columns:
    some_columns_min = min_v[some_columns.index(column)]
    some_columns_max = max_v[some_columns.index(column)]

    if some_norm_data_set[column] < some_columns_min or some_norm_data_set[column] > some_columns_max:
        is_good_data = True

if is_good_data:
    print('выбросы')

else:
    print('Результат алгоритма')
    data = pd.DataFrame([list(some_norm_data_set.values())])
    data.columns = ['gender_male', 'gender_female', 'age', 'height_bad', 'timestamp']
    print(data)
    prediction = knn.predict(data)[0]
    if prediction == 1.:
        print('человек умер')
    else:
        print("человек выжил")
    neigh = NearestNeighbors(n_neighbors=7)
    neigh.fit(df.iloc[:, :-1])
    indexes = neigh.kneighbors(data, return_distance=False)
    print('ближайшие соседи:')
    roommate = df.iloc[indexes[0]]
    print(roommate)
    print('вероятность:',
          len(roommate.loc[(roommate['dead'] == prediction), 'dead']) / 7 * 100, '%')


