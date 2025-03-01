import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple
import numpy as np

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def kmeans(raw_data: np.ndarray, k: int, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    centroids = raw_data[np.random.choice(len(raw_data), k, replace=False)]
    labels = None

    for _ in range(max_iters):
        distances = np.sqrt(((raw_data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([raw_data[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


def main() -> None:
    data = np.concatenate(
        [
            np.random.randn(50, 2) + [-2, 2],
            np.random.randn(50, 2) + [-2, -2],
            np.random.randn(50, 2) + [2, -2],
        ]
    )

    k = 3
    centroids, labels = kmeans(data, k)

    print("聚类中心：\n", centroids)
    print("数据点标签统计：\n", np.bincount(labels))

    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=200, c="r", label='聚类中心')
    plt.title("k均值聚类")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("../img/kmeans_result.png")


if __name__ == "__main__":
    main()
