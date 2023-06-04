import numpy as np
import math, random
import matplotlib.pyplot as plt
import seaborn as sns


def density_estimation(samples: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    density = np.zeros_like(X)
    n = samples.shape[0]
    for i in range(X.shape[0]):
        Rk = np.sort(np.abs(samples - X[i]))[k - 1]
        density[i] = k / (n * 2 * Rk)
    return density


def display(X: np.ndarray, Y: np.ndarray, samples: np.ndarray, k :int) -> None:
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=[8, 5.5], dpi=300)
    ax.plot(X, Y, label=r'$p(x)$')
    if samples.shape[0] <= 16: ax.plot(samples, np.full_like(samples, -0.01), '.r')
    ax.set_title(r'$n = {0},\; k = {1}$'.format(samples.shape[0], k))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p(x)$')
    ax.legend()
    plt.savefig('image_knn1_{0}_{1}.png'.format(samples.shape[0], k), dpi=300)
    plt.show()


def save_data(X: np.ndarray, Y: np.ndarray, fname: str = 'data.txt') -> None:
    with open(file=fname, mode='w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            f.write('{0} {1}\n'.format(round(X[i], 3), round(Y[i], 3)))
    print('Data saved successfully!')


arr_N = [1, 2**5, 2**10, 2**20]
X = np.linspace(-3, 3, 1000)
for N in arr_N:
    k = int(math.sqrt(N))
    # 一维正态分布
    # samples = np.random.normal(loc=0.0, scale=1.0, size=N)
    # 一维混合分布
    samples = np.array([random.uniform(-2, -1.5) if random.random() > 0.5
                        else random.uniform(0, 2) for _ in range(N)])
    density = density_estimation(samples, X, k)
    display(X, density, samples, k)
    save_data(X, density, 'data_knn1_{0}_{1}.txt'.format(N, k))