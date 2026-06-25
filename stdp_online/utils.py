"""
MNIST 加载 / 泊松编码 / 可视化
"""
import numpy as np
import gzip, struct, urllib.request
from pathlib import Path

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download_mnist():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in MNIST_URLS.items():
        fname = url.split("/")[-1]
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  下载 {fname} ...")
            urllib.request.urlretrieve(url, fpath)


def _parse_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _parse_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    """返回 (train_imgs, train_labels, test_imgs, test_labels)"""
    _download_mnist()
    train_imgs = _parse_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_labels = _parse_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    test_imgs = _parse_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = _parse_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    train_imgs = train_imgs.reshape(-1, 784).astype(np.float64) / 255.0
    test_imgs = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0
    return train_imgs, train_labels, test_imgs, test_labels


def poisson_encode(image, duration_ms=350, dt_ms=1.0, max_rate_hz=63.75, seed=None):
    """单张图像 → 泊松脉冲序列 (784, n_steps) bool"""
    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / dt_ms)
    rates = image * max_rate_hz
    prob = rates * (dt_ms / 1000.0)
    return rng.rand(784, n_steps) < prob[:, np.newaxis]


def visualize_weights(weights, neuron_indices, save_path=None):
    """指定神经元的28×28感受野灰度图"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    n = len(neuron_indices)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(neuron_indices):
        axes[i].imshow(weights[idx].reshape(28, 28), cmap="gray", interpolation="nearest")
        axes[i].axis("off")
        axes[i].set_title(f"神经元 {idx}", fontsize=8)
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_label_responses(spike_counts_per_class, assigned_labels, save_path=None):
    """左: 类别分配分布 / 右: 各类别最强响应"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels, counts = np.unique(assigned_labels[assigned_labels >= 0], return_counts=True)
    axes[0].bar(labels, counts)
    axes[0].set_xlabel("数字类别")
    axes[0].set_ylabel("已分配神经元数量")
    axes[0].set_title("神经元类别分配分布")
    axes[0].set_xticks(range(10))

    max_responses = spike_counts_per_class.max(axis=1) if spike_counts_per_class.size > 0 else np.zeros(10)
    axes[1].bar(range(10), max_responses)
    axes[1].set_xlabel("数字类别")
    axes[1].set_ylabel("最大平均发放数")
    axes[1].set_title("各类别最强神经元响应")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
