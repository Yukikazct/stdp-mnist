# STDP脉冲神经网络 — MNIST手写数字识别

**Diehl & Cook (2015) 在线三重STDP — 完整脉冲仿真 + 真实STDP学习规则**

## 快速运行

```bash
cd stdp_spike
pip install -r requirements.txt
python main.py
```

## STDP规则

严格实现 Diehl & Cook (2015) 的三重trace在线STDP：

```
pre trace:  tau=20ms, 输入脉冲发放 → pre=1
post1 trace: tau=20ms, 神经元发放 → post1=1
post2 trace: tau=40ms, 神经元发放 → post2=1

LTD (每个输入脉冲触发):  w[j,i] -= nu_pre × post1[i]      nu_pre=0.0001
LTP (每个神经元发放触发): w[j,i] += nu_post × pre[j] × post2_before[i]  nu_post=0.01
```

权重更新在LIF仿真过程中**在线进行**，每个spike触发，严格依赖spike timing。
权重归一化在每样本前执行（`w *= 78.0 / sum(w)`），作为LTD的全局稳态机制。

## 与 fork 的原始代码对比

本实现 (`stdp_spike/`) vs 论文原始代码 (`stdp-mnist/`):

| | 本实现 (stdp_spike) | 原始代码 (stdp-mnist) |
|------|------|------|
| **仿真器** | Numba JIT (Python) | Brian1 / Brian2 |
| **STDP** | ✅ 在线三重STDP | ✅ 在线三重STDP |
| **LIF神经元** | ✅ 电导型 | ✅ 电导型 |
| **侧向抑制** | ✅ 全局抑制 (17.0) | ✅ E↔I回路 |
| **内在可塑性** | ✅ 自适应theta | ✅ 自适应theta |
| **泊松编码** | ✅ 63.75Hz | ✅ 输入/8 × intensity |
| **权重归一化** | ✅ 每样本 (target=78) | ✅ 每样本 (target=78) |
| **依赖** | numpy + numba + matplotlib | brian2 + numpy |
| **训练速度** | Numba加速 (较慢) | C代码生成 (快) |

## 网络架构

```
输入层 (784 Poisson) → 兴奋层 (400 LIF) ↔ 抑制层 (隐式)
                           ↑
                      在线STDP (LTP + LTD)
```

## 仿真流程

```
每个训练样本:
  1. 权重归一化 (每样本前)
  2. 350ms 刺激期: 泊松脉冲输入 + 在线STDP
  3. 150ms 静息期: 零输入, trace衰减
  4. 输入强度自适应 (发放<5则增加强度)
```

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_EXCITATORY` | 400 | 兴奋神经元数 |
| `DURATION_MS` | 350 | 刺激期(ms) |
| `REST_MS` | 150 | 静息期(ms) |
| `MAX_RATE_HZ` | 63.75 | 泊松编码最大频率 |
| `NU_PRE` | 0.0001 | LTD学习率 |
| `NU_POST` | 0.01 | LTP学习率 |
| `TAU_PRE` | 20 | pre trace时间常数(ms) |
| `TAU_POST1` | 20 | post1 trace时间常数(ms) |
| `TAU_POST2` | 40 | post2 trace时间常数(ms) |
| `W_MAX` | 1.0 | 权重上限 |
| `TARGET_WEIGHT_SUM` | 78.0 | 权重归一化目标 |
| `INH_STRENGTH` | 17.0 | 侧向抑制强度 |
| `N_EPOCHS` | 3 | 训练轮数 |
| `N_TRAIN_SAMPLES` | 60000 | 每轮训练样本数 |
| `TOP_K_VOTE` | 10 | 推理投票神经元数 |

## 生成图表说明

### receptive_fields_stdp.png — 感受野（STDP学习到的特征模板）

每个子图是一个神经元的784维权重reshape为28×28的灰度图。

- **亮区**：神经元对该位置像素敏感，STDP强化了该输入通道
- **清晰数字模板** → 该神经元通过STDP高度特化于识别该类数字
- **模糊模板** → 该神经元被多个类别的样本激活，特征被平均化
- 与Hebbian版对比可看出STDP学习到的特征更稀疏（仅强化时序相关的输入）

### neuron_assignment_stdp.png — 神经元分配与响应统计

**左图（神经元类别分配分布）**：
- 横轴：数字类别 0-9
- 纵轴：被分配到该类别的神经元数量
- 反映STDP训练后各类别"占有"的神经元数量
- **均匀分布** → 内在可塑性（theta自适应）有效平衡了神经元参与度

**右图（各类别最强神经元响应）**：
- 横轴：数字类别 0-9
- 纵轴：该类别中最强神经元的平均发放数
- **柱子越高** → 至少存在一个对该类别高度特化的专家神经元
- 与左图结合可诊断：某类别神经元多但响应弱 → 标签分配质量问题

## 参考

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
- 原始代码: https://github.com/peter-u-diehl/stdp-mnist
