"""
Brian2 STDP 推理 + 可视化

加载预训练的STDP模型，执行三部分操作:
  1. 实时推理 — 100个样本逐张预测，展示在线推理速度
  2. 完整准确率 — 从预存的confusion.npy直接读取10K测试集结果
  3. 可视化 — 生成感受野图 + 神经元分配统计图


"""
import os, numpy as np, time
from pathlib import Path

# ── 路径配置 ──
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data' / 'stdp_full'          # 预训练模型和结果目录
OUT_DIR = HERE                                   # 可视化输出目录
N_LIVE = 100                                     # 实时推理样本数

# ── Brian2初始化 ──
os.environ['CFLAGS'] = '-O3'
from brian2 import prefs
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

import stdp_model as dc
dc.MNIST_PATH = HERE / '..' / 'data'
dc.DATA_PATH = DATA_DIR

# ── 加载预训练模型 ──
# assign.npy: 训练后observe阶段生成的标签分配表
# 每个神经元被分配到其最常响应的数字类别
assign = np.load(str(DATA_DIR / 'assign.npy'))
groups = [np.where(assign == i)[0] for i in range(10)]   # 各类别神经元索引列表

X_test, Y_test = dc.read_mnist(False)                     # 加载测试集（仅评估用）
dc.N_NEURONS = len(assign)                                # 从assign推断神经元数
net = dc.build_network(False)                             # 推理模式（STDP关闭）


# 1. 实时推理 — 100个样本逐张预测
print(f"实时推理 {N_LIVE} 样本...")
t0 = time.time()
correct = 0
for i in range(N_LIVE):
    # 每张图像: 350ms仿真 → 各神经元发放计数
    exc = dc.show_sample(net, X_test[i], dc.INTENSITY)
    # 预测: 每组(类别)神经元的平均发放率 → 选最高组
    guess = np.argmax([exc[grp].mean() for grp in groups])
    if guess == Y_test[i]:
        correct += 1
live_acc = correct / N_LIVE * 100
print(f"实时: {correct}/{N_LIVE} = {live_acc:.1f}% ({time.time() - t0:.1f}s)")

# 2. 完整准确率 — 从预存的混淆矩阵直接读取
conf = np.load(str(DATA_DIR / 'confusion.npy'))
full_acc = np.trace(conf) / np.sum(conf) * 100
print(f"\n★ STDP准确率: {full_acc:.2f}% (10K)")
for i in range(10):
    row = conf[i]
    a = row[i] / row.sum() * 100 if row.sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# 3. 可视化
from utils import visualize_weights, visualize_label_responses

# 3a. 感受野: 从每类选2个神经元，展示其784维权重重塑为28×28灰度图

w_ie = np.load(str(DATA_DIR / 'weights.npy')).reshape(len(assign), -1)
idxs = []
for c in range(10):
    c_neurons = np.where(assign == c)[0]
    if len(c_neurons) >= 2:
        idxs.extend(c_neurons[:2])               # 每类取前2个
if idxs:
    visualize_weights(w_ie, idxs, save_path=str(OUT_DIR / 'receptive_fields_stdp.png'))

# 3b. 神经元分配: 各类别神经元数量和最强响应
expanded = np.zeros((10, len(assign)))
for c in range(10):
    expanded[c, assign == c] = 1
visualize_label_responses(expanded, assign, save_path=str(OUT_DIR / 'neuron_assignment_stdp.png'))
