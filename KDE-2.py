import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random, math


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    vec = np.stack([X, Y], axis=2)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = vec - mean
    exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_cov, diff)
    return 1.0 / (2 * np.pi * np.sqrt(det_cov)) * np.exp(exponent)


def density_estimation(X: np.ndarray, Y: np.ndarray, samples: np.ndarray, window_size: np.ndarray) -> np.ndarray:
    density = np.zeros_like(X)
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    for sample in samples:
        _X = (X - sample[0]) / window_size[0]
        _Y = (Y - sample[1]) / window_size[1]
        density += gaussian_kernel(_X, _Y, mean, cov)
    return density / (samples.shape[0] * np.prod(window_size))


def save_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, fname: str = 'data.txt') -> None:
    with open(file=fname, mode='w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write('{0} {1} {2}\n'.format(round(X[i, j], 3), round(Y[i, j], 3), round(Z[i, j], 3)))
            f.write('\n')
    print('Data saved successfully!')


def display(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, samples: np.ndarray, h1: float) -> None:
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$p(x,\; y)$')
    ax.set_title(r'$n = {0},\; h_1 = {1}$'.format(samples.shape[0], h1))
    plt.savefig('image_kde2_{0}_{1}.png'.format(samples.shape[0], h1), dpi=300)
    plt.show()


arr_N = [1, 2**5, 2**10, 2**20]
arr_h1 = [0.25, 0.5, 1.0]
X = np.linspace(-3, 3, 50)
Y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(X, Y)
for N in arr_N:
    # 二维正态分布
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    samples = np.random.multivariate_normal(mean, cov, N)
    # 二维混合分布
    # samples = np.array([
    #     [random.uniform(0, 2), random.uniform(0, 2)] if random.random() > 0.5
    #     else [random.uniform(-2, -1), random.uniform(-2, -1)] for _ in range(N)
    # ])
    for h1 in arr_h1:
        window_size = np.array([h1 / math.pow(N, 0.25), h1 / math.pow(N, 0.25)])
        density = density_estimation(X, Y, samples, window_size)
        save_data(X, Y, density, 'data_kde2_{0}_{1}.txt'.format(N, h1))
        display(X, Y, density, samples, h1)