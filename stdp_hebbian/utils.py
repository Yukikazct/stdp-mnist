"""
数据加载与脉冲编码工具
- MNIST数据集加载（自动下载）
- 泊松脉冲序列编码
- 权重可视化工具
"""

import numpy as np
import gzip
import struct
import os
import urllib.request
from pathlib import Path

# MNIST数据URL
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).parent.parent / "data"  # 共享数据目录


def _download_mnist():
    """下载MNIST数据集到本地data目录"""
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in MNIST_URLS.items():
        fname = url.split("/")[-1]
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  下载 {fname} ...")
            urllib.request.urlretrieve(url, fpath)
            print(f"  完成: {fname}")


def _parse_images(filepath):
    """解析MNIST图像文件，返回 (N, 28, 28) uint8数组"""
    with gzip.open(filepath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data


def _parse_labels(filepath):
    """解析MNIST标签文件，返回 (N,) uint8数组"""
    with gzip.open(filepath, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist():
    """
    加载MNIST数据集，如本地不存在则自动下载。
    返回: (train_imgs, train_labels, test_imgs, test_labels)
      train_imgs: (60000, 784) float64, 归一化到[0,1]
      train_labels: (60000,) int
      test_imgs:  (10000, 784) float64
      test_labels:  (10000,) int
    """
    _download_mnist()

    train_imgs = _parse_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_labels = _parse_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    test_imgs = _parse_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = _parse_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    # 展平并归一化到 [0, 1]
    train_imgs = train_imgs.reshape(-1, 784).astype(np.float64) / 255.0
    test_imgs = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0

    return train_imgs, train_labels, test_imgs, test_labels


def poisson_encode(image, duration_ms=350, dt_ms=1.0, max_rate_hz=63.75, seed=None):
    """
    将单张图像编码为泊松脉冲序列。

    参数:
        image: (784,) 像素值 ∈ [0, 1]
        duration_ms: 仿真时长 (ms)
        dt_ms: 时间步长 (ms)
        max_rate_hz: 最大发放率 (Hz), 对应像素值=1.0
        seed: 随机种子

    返回:
        spikes: (784, n_steps) bool数组, True=发放脉冲
    """
    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / dt_ms)
    rates = image * max_rate_hz  # (784,) Hz
    prob = rates * (dt_ms / 1000.0)  # 每个时间步的发放概率
    spikes = rng.rand(784, n_steps) < prob[:, np.newaxis]
    return spikes


def visualize_weights(weights, neuron_indices, save_path=None):
    """
    将指定神经元的输入权重可视化为28×28图像。

    参数:
        weights: (n_excitatory, 784) 权重矩阵
        neuron_indices: 要可视化的神经元索引列表
        save_path: 保存路径 (可选)
    """
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
        w = weights[idx].reshape(28, 28)
        axes[i].imshow(w, cmap="gray", interpolation="nearest")
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
    """
    可视化每个兴奋神经元对各类别的响应及标签分配。

    参数:
        spike_counts_per_class: (10, n_excitatory) 每个神经元对各类别的平均发放数
        assigned_labels: (n_excitatory,) 分配的标签, -1表示未分配
        save_path: 保存路径 (可选)
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 标签分配分布
    labels, counts = np.unique(assigned_labels[assigned_labels >= 0], return_counts=True)
    axes[0].bar(labels, counts)
    axes[0].set_xlabel("数字类别")
    axes[0].set_ylabel("已分配神经元数量")
    axes[0].set_title("神经元类别分配分布")
    axes[0].set_xticks(range(10))

    # 右图: 每个类别的最大响应
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
