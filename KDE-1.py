import numpy as np
import matplotlib.pyplot as plt
import random, math
import seaborn as sns


def gaussian_kernel(X: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    exponent = -0.5 * ((X - mu) / sigma) ** 2
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(exponent)


def density_estimation(X: np.ndarray, samples: np.ndarray, window_size: np.ndarray) -> np.ndarray:
    density = np.zeros_like(X)
    mu, sigma = 0.0, 1.0
    for sample in samples:
        _X = (X - sample) / window_size[0]
        density += gaussian_kernel(_X, mu, sigma)
    return density / (samples.shape[0] * np.prod(window_size))


def save_data(X: np.ndarray, Y: np.ndarray, fname: str = 'data.txt') -> None:
    with open(file=fname, mode='w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            f.write('{0} {1}\n'.format(round(X[i], 3), round(Y[i], 3)))
    print('Data saved successfully!')


def display(X: np.ndarray, Y: np.ndarray, samples: np.ndarray, h1: float) -> None:
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=[8, 5.5], dpi=300)
    ax.plot(X, Y, label=r'$p(x)$')
    # 当样本点数量较少时，在图中标记出样本点
    if samples.shape[0] <= 16: ax.plot(samples, np.full_like(samples, -0.01), '.r')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p(x)$')
    ax.set_title(r'$N = {0},\; h_1 = {1}$'.format(samples.shape[0], h1))
    plt.legend()
    plt.savefig('image_kde1_{0}_{1}.png'.format(samples.shape[0], h1), dpi=300)
    plt.show()


arr_N = [1, 2**5, 2**10, 2**20]
arr_h1 = [0.25, 0.5, 1.0]
X = np.linspace(-3, 3, 1000)
for N in arr_N:
    # 一维正态分布
    samples = np.random.normal(loc=0.0, scale=1.0, size=N)
    # 一维混合分布
    # samples = np.array([random.uniform(-2.5, -2) if random.random() > 0.5
    #                     else random.uniform(0, 2) for _ in range(N)])
    for h1 in arr_h1:
        window_size = np.array([h1 / math.sqrt(N)])
        density = density_estimation(X, samples, window_size)
        save_data(X, density, 'data_kde1_{0}_{1}.txt'.format(N, h1))
        display(X, density, samples, h1)