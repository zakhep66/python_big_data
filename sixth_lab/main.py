import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pandas.read_csv('../second_lab/normalized_data_set.csv')

death_bad = pandas.read_csv('../second_lab/normalized_data_set.csv')
parameters = ['gender_male', 'gender_female', 'age', 'height_bad', 'timestamp', 'dead']
death_bad = death_bad[parameters]

# разбиваем на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(
    death_bad.iloc[:, :-1],
    death_bad['dead'],
    test_size=0.25,
    random_state=3
)
print(X_test)
print(death_bad.iloc[:, :-1])

# классификатор

classifier = svm.SVC(probability=True)
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

contrast = classifier.predict(X_test)  # вывод разницы в процентах

print('правильность: ', accuracy_score(y_test, contrast))

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

# проверяем только что сгенеренный dataset на выбросы
min_v = [
    death_bad['gender_male'].mean() - 3 * death_bad['gender_male'].std(),
    death_bad['gender_female'].mean() - 3 * death_bad['gender_female'].std(),
    death_bad['age'].mean() - 3 * death_bad['age'].std(),
    death_bad['height_bad'].mean() - 3 * death_bad['height_bad'].std(),
    death_bad['timestamp'].mean() - 3 * death_bad['timestamp'].std()
]  # минус три сигма

max_v = [
    death_bad['gender_male'].mean() + 3 * death_bad['gender_male'].std(),
    death_bad['gender_female'].mean() + 3 * death_bad['gender_female'].std(),
    death_bad['age'].mean() + 3 * death_bad['age'].std(),
    death_bad['height_bad'].mean() + 3 * death_bad['height_bad'].std(),
    death_bad['timestamp'].mean() + 3 * death_bad['timestamp'].std()
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
    data = pandas.DataFrame([list(some_norm_data_set.values())])
    data.columns = ['gender_male', 'gender_female', 'age', 'height_bad', 'timestamp']
    print(data)
    prediction = classifier.predict(data)[0]
    if prediction == 1.:
        print("человек умер")
    else:
        print('человек выжил')
