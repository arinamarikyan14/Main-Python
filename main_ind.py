# ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ
# Блок 1. Подключение библиотек
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import json

# Блок 2. Генерация массива M по варианту 4
np.random.seed(0)
n_samples = 200

var1 = np.random.uniform(-10, 1, n_samples)
var2 = np.random.uniform(1, 2, n_samples)
var3_text = np.random.choice(
    ["отрицательное значение", "положительное значение"], n_samples
)


# Блок 3. Преобразование категориального признака
mapping = {"отрицательное значение": 0, "положительное значение": 1}

var3 = [mapping[value] for value in var3_text]


# Блок 4. Формирование таблицы данных
df = pd.DataFrame({"var1": var1, "var2": var2, "var3": var3})

print("Исходный массив M, первые 5 строк:")
print(df.head(5))

# Блок 5. Поиск числа кластеров методом локтя
inertias = []
k_range = range(1, 12)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    model.fit(df)
    inertias.append(model.inertia_)


# Блок 6. Построение графика метода локтя
plt.figure()
plt.plot(list(k_range), inertias, "bo-")
plt.xlabel("Количество кластеров k")
plt.ylabel("Внутрикластерная сумма квадратов")
plt.title("Метод локтя для варианта 4")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_var4.png")
plt.close()


# Блок 7. Обучение итоговой модели KMeans
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0, n_init=10)

kmeans.fit(df)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
final_inertia = kmeans.inertia_
iterations = kmeans.n_iter_
cluster_sizes = Counter(labels)

# Блок 8. Вывод результатов кластеризации
print("Прогнозируемые кластеры для каждой записи массива M:")
print(labels)

print("Координаты центроидов кластеров:")
print(centroids)

print(f"Внутрикластерная сумма квадратов итоговой модели: {final_inertia:.4f}")
print(f"Количество итераций KMeans: {iterations}")

print("Размер каждого кластера:")
print(cluster_sizes)

# Блок 9. Сохранение результатов в JSON
results = {
    "variant": 4,
    "dataset_head": df.head(5).to_dict(orient="records"),
    "selected_clusters": 4,
    "labels": labels.tolist(),
    "centroids": centroids.tolist(),
    "final_inertia": final_inertia,
    "iterations": iterations,
    "cluster_sizes": {
        str(cluster): int(size) for cluster, size in cluster_sizes.items()
    },
    "mapping": mapping,
}

with open("results_var4.json", "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

# Блок 10. Построение 3D-диаграммы рассеяния
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["var1"], df["var2"], df["var3"], c=labels, s=40)
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    marker="X",
    s=120,
    label="Центроиды",
)

ax.set_xlabel("var1")
ax.set_ylabel("var2")
ax.set_zlabel("var3")
ax.set_title("Диаграмма рассеяния KMeans для варианта 4")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_var4.png")
plt.close()