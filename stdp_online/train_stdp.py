"""
STDP完整训练 — Diehl & Cook (2015) 无监督 STDP 学习

三阶段流水线:
  1. train()   — STDP无监督训练，180K样本（60K×3轮），~12.5小时
  2. observe() — 标签分配，用5K样本统计神经元类别偏好
  3. test()    — 测试评估，10K测试集，生成混淆矩阵

输出目录 (data/stdp_full/):
  weights.npy      — 输入→兴奋 STDP权重 (784×1200)
  assign.npy       — 神经元→类别分配表 (1200个整数标签)
  theta.npy        — 最终自适应阈值
  confusion.npy    — 10×10混淆矩阵
  train_stats.npy  — 训练过程中的统计序列
  train_w_hist.npy — 权重历史快照

Brian2加速策略:
  - 使用Cython C代码生成后端（而非纯Python）
  - GCC编译优化: -O3 -ffast-math
  - 这些优化将Python仿真速度提升到接近C级别

用法:
  python train_stdp.py     # 完整训练（约12.5小时）
"""
import os, time, numpy as np
from pathlib import Path

# GCC编译优化:
os.environ['CFLAGS'] = '-O3'

# ── 项目路径 ──
HERE = Path(__file__).resolve().parent       # stdp_online/ 目录

# ── Brian2初始化 ──
from brian2 import prefs, seed as bseed
# 设置C++编译选项，生成高效Cython代码
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'            

import stdp_model as dc                      # STDP核心模型


# 超参数设置

dc.MNIST_PATH = HERE / '..' / 'data'         # 共享MNIST数据目录
dc.DATA_PATH = HERE / 'data' / 'stdp_full'   # 本方法输出目录
dc.DATA_PATH.mkdir(parents=True, exist_ok=True)

dc.N_NEURONS = 1200   
dc.N_TRAIN = 180000     
dc.N_OBSERVE = 5000    
dc.N_TEST = 10000       
dc.SEED = 42            

# 固定所有随机源
bseed(42)
np.random.seed(42)

print("=" * 50)
print("STDP: 1200神经元 × 180K样本")
print("=" * 50)

# ── 阶段1: STDP无监督训练 ──
t0 = time.time()
dc.train()
print(f"训练: {(time.time() - t0) / 60:.0f}min")

# ── 阶段2: 标签分配 ──
t0 = time.time()
dc.observe()
print(f"标签: {(time.time() - t0) / 60:.0f}min")

# ── 阶段3: 测试评估 ──
t0 = time.time()
dc.test()
print(f"测试: {(time.time() - t0) / 60:.0f}min")
print("完成!")
