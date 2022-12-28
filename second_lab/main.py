import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# df = pd.read_csv('data_set.csv')
#
# # удаление нулевых полей
# ##########################################
#
# df = df[df['Close'].notna()]
# df.to_csv('result.csv', index_label=False, index=False)  # запись dataframe без нулевых строк

##########################################

# Импортируем датасет
df = pd.read_csv('data_set.csv')
print(df)
print(df.dtypes)

# Перевод качественных признаков типа колонки в количественный
full_count_data_set = pd.get_dummies(df, columns=['gender'])
print(full_count_data_set)

normalized_dataset = full_count_data_set[["gender_male", "gender_female", "age", "height_bad", "timestamp", "dead"]]
print("\nВсего пустот:\n", full_count_data_set.isnull().sum())


# Тренировка в уравнивании классов
# low = normalized_dataset[normalized_dataset["dead"] == 0]  # удаление случайных строк с dead == 0
# to_remove = low.sample(n=400).index.tolist()  # вместо ... нужно указать количество строк
# normalized_dataset.drop(to_remove, inplace=True)
# print(low)

# Инициализация нормализатора
scaler = MinMaxScaler()

# Передача датасета и преобразование
scaler.fit(normalized_dataset)
normalized_dataset = scaler.transform(normalized_dataset)

normalized_dataset = pd.DataFrame(data=normalized_dataset,
                                  columns=["gender_male", "gender_female", "age", "height_bad", "timestamp", "dead"],
                                  index=None)

print("normalized_dataset:\n", normalized_dataset)
normalized_dataset.to_csv('normalized_data_set.csv', index_label=False, index=False)

# Проверка на выбросы
dead_boxplot = df.boxplot(column=['dead'])
boxplot = df['dead'].describe()
print(boxplot)
boxplot = df['dead'].value_counts().plot.bar()
plt.show()

# Гистограмма по dead
df['dead'].plot(kind='hist', density=1, bins=20, stacked=False, alpha=.5, color='red')
plt.show()

# Выбросы

columns = ["gender_male", "gender_female", "age", "height_bad", "timestamp"]

for column in columns:
    out = normalized_dataset.loc[
        (normalized_dataset[column] < (normalized_dataset[column].mean() - 3 * normalized_dataset[column].std())) | (
                    normalized_dataset[column] > (
                        normalized_dataset[column].mean() + 3 * normalized_dataset[column].std())), column]
    print(out)
    out_indexes = normalized_dataset.index[
        (normalized_dataset[column] < (normalized_dataset[column].mean() - 3 * normalized_dataset[column].std())) | (
                    normalized_dataset[column] > (
                        normalized_dataset[column].mean() + 3 * normalized_dataset[column].std()))].tolist()
    print(out_indexes)
    normalized_dataset.drop(normalized_dataset.index[out_indexes], axis=0, inplace=True)
print(normalized_dataset)
