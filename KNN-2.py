import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def density_estimation(X: np.ndarray, Y: np.ndarray, samples: np.ndarray, k: int) -> np.ndarray:
    points = np.stack([X, Y], axis=2)
    density = np.zeros_like(X)
    n = samples.shape[0]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Rk = np.sort(np.sum((samples - points[i, j]) ** 2, axis=1))[k - 1]
            density[i, j] = k / (n * np.pi * Rk)
    return density


def save_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, fname: str = 'data.txt') -> None:
    with open(file=fname, mode='w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write('{0} {1} {2}\n'.format(round(X[i, j], 3), round(Y[i, j], 3), round(Z[i, j], 3)))
            f.write('\n')
    print('Data saved successfully!')


def display(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, samples: np.ndarray, k: int) -> None:
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$p(x,\; y)$')
    ax.set_title(r'$n = {0}, k = {1}$'.format(samples.shape[0], k))
    plt.savefig('image_knn2_{0}_{1}.png'.format(samples.shape[0], k), dpi=300)
    plt.show()


arr_N = [1, 2**5, 2**10, 2**20]
X = np.linspace(-3, 3, 50)
Y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(X, Y)
for N in arr_N:
    k = int(math.sqrt(N))
    # 二维正态分布
    # mean = np.array([0.0, 0.0])
    # cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    # samples = np.random.multivariate_normal(mean, cov, N)
    # 二维混合分布
    samples = np.array([
        [random.uniform(0, 2), random.uniform(0, 2)] if random.random() > 0.5
        else [random.uniform(-2, -1), random.uniform(-2, -1)] for _ in range(N)
    ])
    density = density_estimation(X, Y, samples, k)
    save_data(X, Y, density, 'data_knn2_{0}_{1}.txt'.format(N, k))
    display(X, Y, density, samples, k)