# ОБЩЕЕ ЗАДАНИЕ
'''# БЛОК 0. ИСХОДНЫЙ ГРАФИК MATPLOTLIB
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# Строится простой линейный график по двум спискам значений.
ax.plot([1, 2, 3, 4] [1, 4, 2, 5])
plt.ylabel('some numbers')
plt.savefig('fig')'''


# БЛОК 1. ИМПОРТ БИБЛИОТЕК ДЛЯ ГЕНЕРАЦИИ И ОБРАБОТКИ ДАННЫХ
# В этом блоке подключаются библиотеки, необходимые для создания набора данных и представления его в табличном виде.
from sklearn.datasets import make_blobs
import pandas as pd

# БЛОК 2. ГЕНЕРАЦИЯ ИСХОДНОГО НАБОРА ДАННЫХ

dataset, classes = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=0.5,
    random_state=0
)

df = pd.DataFrame(dataset, columns=['var1', 'var2'])
# В консоль выводятся первые две строки набора данных.
print(df.head(2))

# БЛОК 3. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ
# В этом блоке используется метод локтя.

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Создается модель KMeans с 4 кластерами.
model = KMeans(n_clusters=4, n_init=10, random_state=0)
visualizer = KElbowVisualizer(model, k=(2, 12), force_model=True)
visualizer.fit(df)
# График метода локтя сохраняется в файл elbow_plot.png.
visualizer.show(outpath="elbow_plot.png")

# БЛОК 4. ОБУЧЕНИЕ МОДЕЛИ KMEANS
# В этом блоке выполняется кластеризация данных методом K-средних.
import matplotlib.pyplot as plt

# Создается новое поле для построения графиков.
plt.figure()
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    random_state=0
).fit(df)


# БЛОК 5. ВЫВОД РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ В КОНСОЛЬ
# Выводятся прогнозируемые метки кластеров для каждой точки данных.
print(kmeans.labels_)
# Выводятся координаты центроидов каждого кластера.
print(kmeans.cluster_centers_)
# Выводится внутрикластерная сумма квадратов.
print(kmeans.inertia_)
# Выводится количество итераций, которое потребовалось алгоритму KMeans.
print(kmeans.n_iter_)


# БЛОК 6. ПОДСЧЕТ РАЗМЕРА КАЖДОГО КЛАСТЕРА
from collections import Counter
# Подсчитывается количество точек с каждой меткой кластера.
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))


# БЛОК 7. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ
# В этом блоке строится диаграмма рассеяния.
import seaborn as sns

# Строится диаграмма рассеяния по признакам var1 и var2.
sns.scatterplot(
    data=df,
    x="var1",
    y="var2",
    hue=kmeans.labels_
)
plt.savefig('clusters.png')


# БЛОК 8. ДОБАВЛЕНИЕ ЦЕНТРОИДОВ НА ГРАФИК
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    c="r",
    s=80,
    label='Centroids'
)
# Добавляется легенда для обозначения центроидов.
plt.legend()
plt.savefig('clusters_with_centroids.png') 