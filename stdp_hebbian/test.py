"""
Hebbian模型测试 — 实时推理 + 混淆矩阵验证

分三部分:
  1. 实时推理 — 加载训练好的Hebbian模型, 对100个样本逐张预测并统计准确率
  2. 完整准确率 — 从预存的confusion_hebbian.npy读取混淆矩阵
  3. 可视化 — 生成感受野图和神经元分配图
"""
import sys, os, numpy as np, time
from utils import load_mnist, visualize_weights, visualize_label_responses
from model import SNN

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(MODEL_DIR, "confusion_hebbian.npy")      # 混淆矩阵 (由 main.py test_pipeline 生成)
npz_path = os.path.join(MODEL_DIR, "model_hebbian.npz")            # Hebbian训练模型

if not os.path.exists(npz_path):
    print("❌ model_hebbian.npz 不存在, 先运行 main.py 训练")
    sys.exit(1)

# ---- 1. 实时推理: 加载模型, 逐样本预测并统计准确率 ----
N_LIVE = 100
# 只加载测试集 (训练集不需要)
_, _, test_imgs, test_labels = load_mnist()
snn = SNN.load(npz_path)

print(f"实时推理 {N_LIVE} 样本...")
t0 = time.time()
correct = 0
for i in range(N_LIVE):
    # predict(): 前向推理 → Top-K发放神经元投票 → 预测类别
    pred, _ = snn.predict(test_imgs[i], top_k=10, seed=700000 + i)
    if pred == test_labels[i]:
        correct += 1
live_acc = correct / N_LIVE * 100
print(f"实时: {correct}/{N_LIVE} = {live_acc:.1f}% (耗时 {time.time()-t0:.1f}s)")

# ---- 2. 完整准确率: 从预存混淆矩阵直接读取 (10K测试集, 秒出结果) ----
if os.path.exists(conf_path):
    conf = np.load(conf_path)
    # 对角元素 = 各类别正确分类数, 总和 = 各类别总数
    full_acc = np.trace(conf) / np.sum(conf) * 100
    print(f"\n★ Hebbian准确率: {full_acc:.2f}% (10K测试集)")
    for i in range(10):
        row = conf[i]
        a = row[i] / row.sum() * 100 if row.sum() > 0 else 0
        print(f"  {i}: {a:.1f}%")
else:
    print(" 混淆矩阵不存在")

# ---- 3. 可视化: 感受野 + 神经元标签分配 ----
# 3a. 感受野: 从每类选响应最强的2个神经元, 展示其输入权重 (28×28灰度图)
idxs = []
for c in range(10):
    c_neurons = np.where(snn.assigned_labels == c)[0]
    if len(c_neurons) >= 2 and snn.spike_counts_per_class is not None:
        # 按该类投票数排序, 取投给该类别票数最多的2个神经元
        best = c_neurons[np.argsort(snn.spike_counts_per_class[c, c_neurons])[-2:]]
        idxs.extend(best)
if idxs:
    visualize_weights(snn.w_ie, idxs, save_path=os.path.join(MODEL_DIR, "receptive_fields_hebbian.png"))
# 3b. 神经元分配: 左图=各类别神经元数量, 右图=各类别最强神经元响应值
if snn.spike_counts_per_class is not None:
    visualize_label_responses(snn.spike_counts_per_class, snn.assigned_labels,
                              save_path=os.path.join(MODEL_DIR, "neuron_assignment_hebbian.png"))
print("图片已更新")
