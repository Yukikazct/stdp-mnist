# 基于STDP的手写数字识别

MNIST手写数字识别 | 脉冲神经网络 | 无监督STDP学习

---

## 目录

1. [最终结果](#最终结果)
2. [设计思路：为什么这样设计](#设计思路为什么这样设计)
3. [项目架构](#项目架构)
4. [方法一：在线三元STDP (Brian2)](#方法一在线三元stdp-brian2)
5. [方法二：STDP速率近似 / Hebbian竞争学习 (Numba)](#方法二stdp速率近似--hebbian竞争学习-numba)
6. [训练集与测试集分割](#训练集与测试集分割)
7. [生成图表说明](#生成图表说明)
8. [环境配置](#环境配置)
9. [运行指南](#运行指南)
10. [常见问题](#常见问题)
11. [总结](#总结)
12. [参考](#参考)

---

## 最终结果

| 方法 | 神经元 | 训练量 | 准确率 | 训练耗时 |
|------|------|------|------|------|
| **在线三元STDP** | 1200 | 180K | **92.39%** | ~12.5h |
| **STDP速率近似 (Hebbian)** | 3000 | 240K | **91.90%** | 17min |

---

## 设计思路：为什么这样设计

### 核心问题

用脉冲神经网络做手写数字识别，有两条路线：

| 路线 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **纯脉冲在线STDP** | 每个spike实时触发权重更新，严格依赖脉冲时序 | 生物放在 | 计算极慢（需C代码生成加速） |
| **速率近似** | 仿真后根据发放率一次性更新权重 | 训练快100倍 | 是否等效于STDP？需要验证 |

**本项目的核心目的：同时实现两条路线，验证它们是否等效。**

### 为什么STDP可以近似为Hebbian + 权重归一化？

Zenke等人(2015)从数学上证明：当STDP的LTP和LTD到达**稳态平衡**时，权重向量的方向收敛到输入模式的聚类中心。

在稳态下：
- STDP的**LTP** (pre-before-post) → 权重向输入模式靠拢 → **等价于 Hebbian LTP**
- STDP的**LTD** (post-before-pre) → 未增强的权重相对衰减 → **等价于 权重归一化**

这意味着：如果你不关心精确的spike timing，只关心最终学到的特征，你完全可以用更快的速率近似代替在线STDP。

**两种方法的对比就是对这个理论的实验验证。**

### 为什么用LIF神经元？

Leaky Integrate-and-Fire (LIF) 是最简单的脉冲神经元模型，它在计算效率和生物学合理性之间取得最佳平衡：

- 比 IF (无泄漏) 更真实 — 膜电位会衰减，不会无限积累
- 足够捕捉 SNN 的核心动态：**时序编码**和**发放率编码**

### 为什么用电导型突触（conductance-based）而不是电流型（current-based）？

```
电导型: I_syn = ge · (-v)           ← 突触电流依赖膜电位
电流型: I_syn = ge                   ← 突触电流与膜电位无关
```

电导型突触有两个关键优势：
1. **分流抑制（shunting inhibition）**：当抑制输入到来时，膜电位越接近静息电位，抑制效果越强 — 更符合生物学
2. **自稳定性**：膜电位越高，兴奋性驱动力越小（因为 `E_exc - v` 变小），自然防止过度兴奋

### 为什么需要侧向抑制（E↔I回路）？

没有侧向抑制，所有神经元会对同一输入同时发放 → 学到的特征都一样 → 没有竞争 → 没有特化。

```
兴奋→抑制→兴奋 回路:
  某兴奋神经元发放 → 激活对应的抑制神经元 → 抑制所有其他兴奋神经元
  → 只有一个或少数几个神经元获胜 → 实现涌现式 WTA (Winner-Take-All)
```

这是**无监督竞争学习**的核心机制。Diehl & Cook (2015) 使用的 E↔I 一对一连接 + I↔E 全连接（除自己）结构，让抑制强度自适应 — 发放越多，抑制越强。

### 为什么需要自适应阈值（内在可塑性）？

```
每次发放后: theta += 0.05mV  (阈值升高，更难再发放)
不发时:     theta 缓慢衰减到 20mV
```

没有自适应阈值 → 某些神经元权重初始化更优 → 一开始就频繁获胜 → STDP继续强化 → 垄断 → 其他神经元无法学习。

自适应阈值确保每个神经元都有参与竞争的机会，实现**负载均衡**。

### 为什么用三重trace STDP？

Diehl & Cook (2015) 的三重trace机制：

```
pre trace:    τ=20ms   → 输入脉冲到达 → pre=1   → 用于计算LTP
post1 trace:  τ=20ms   → 神经元发放   → post1=1 → 用于计算LTD
post2 trace:  τ=40ms   → 神经元发放   → post2=1 → 用于调制LTP
```

为什么需要三个trace而不是两个（标准STDP只有pre和post）？

- **post1 (τ=20ms)**：负责LTD — 短时间窗口确保只有最近的发放才触发抑制
- **post2 (τ=40ms)**：负责LTP的调制 — 更长的窗口允许更宽的时间关联范围
- 引入 `post2bef`（发放前的post2值）意味着LTP发生在 pre-before-post 情况下

这种设计增加了稳定性：LTD（post1）和LTP调制（post2）使用不同时间常数，避免权重在同一个发放事件中既增强又抑制。

### 为什么采用三阶段流水线（train → observe → test）？

这是**无监督学习 + 事后标签分配**的标准流程：

1. **train()** — 无监督STDP训练：权重更新只依赖脉冲时序，完全不使用标签
2. **observe()** — 标签分配阶段：冻结权重，用训练集前N个样本统计每个神经元对各类别的平均响应，投票决定每个神经元"代表"哪个数字
3. **test()** — 测试阶段：冻结权重和标签分配，用测试集评估准确率

训练和标签分配分离的原因：**STDP是无监督的** — 权重更新规则中没有类别信息。网络自发形成对输入模式的聚类，事后我们才"标注"每个神经元代表了什么。

### 为什么用泊松编码？

MNIST是静态图像，没有时间维度。要送入脉冲神经网络，必须把像素值转换为脉冲序列。

泊松编码是最自然的选择：
```
每个像素 i 在每个时间步 t:
  发放概率 = pixel_value[i] × max_rate × dt / 1000
  实际发放 = Bernoulli(发放概率)
```

- **生物学合理**：真实神经元的发放近似泊松过程
- **实现简单**：无需复杂的编码器
- **时间信息丰富**：每张图像产生不同的脉冲序列模式，STDP可以捕捉时序相关性

### 为什么同时需要Brian2和Numba两种实现？

| 需求 | Brian2 | Numba |
|------|--------|-------|
| **在线STDP** |  内置支持（每个spike触发代码） |  需手动实现 |
| **C代码生成** |  Cython C扩展 |  JIT编译（轻量） |
| **仿真速度** | C级别（需编译） | C级别（运行时编译） |
| **灵活性** | 受限于Brian2框架 | 完全自定义 |
| **适用场景** | 真实STDP仿真 | 速率近似 |

- Brian2版本：严格复现论文的细节
- Numba版本：快速验证STDP稳态理论的预测

两者互补：Brian2验证正确性，Numba验证速率近似理论的正确性，同时也证明了速率近似在实际应用中的可行性。

---

## 项目架构

```
基于STDP的手写数字识别/
│
├── README.md                    # 项目总说明（本文件）
│
├── data/                        # MNIST原始数据集（各版本共享）
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── stdp_online/                 # 方法一：在线三元STDP（Brian2 + Cython加速）
│   ├── stdp_model.py            #   STDP核心模型 (LIF神经元 + 三重trace + E↔I回路)
│   ├── train_stdp.py            #   完整训练脚本 (超参数设置 + 三阶段调用)
│   ├── test_brian.py            #   推理测试 + 可视化 (感受野 + 分类准确率)
│   ├── utils.py                 #   MNIST下载/解析 + 泊松编码 + 可视化工具
│   ├── requirements.txt         #   brian2, numpy, matplotlib, progressbar2
│   ├── README.md                #   方法一详细说明
│   ├── data/stdp_full/          #   训练输出 (weights / assign / theta / confusion)
│   ├── receptive_fields_stdp.png
│   └── neuron_assignment_stdp.png
│
└── stdp_hebbian/                # 方法二：STDP速率近似 (Numba JIT加速)
    ├── model.py                 #   Hebbian竞争学习模型 (LIF仿真 + 赢家更新 + 权重归一化)
    ├── main.py                  #   完整训练脚本
    ├── test.py                  #   推理测试 + 可视化
    ├── utils.py                 #   可视化工具 (感受野 + 神经元分配)
    ├── requirements.txt         #   numpy, numba, matplotlib, tqdm
    ├── README.md                #   方法二详细说明
    ├── model_hebbian.npz        #   已训练模型
    ├── confusion_hebbian.npy    #   混淆矩阵
    ├── receptive_fields_hebbian.png
    └── neuron_assignment_hebbian.png
```

### 架构分层示意

```
┌─────────────────────────────────────────────────────────┐
│                      应用层                              │
│   train_stdp.py / test_brian.py / test.py / main.py     │
│   (超参数配置, 流程编排, 结果可视化)                       │
├─────────────────────────────────────────────────────────┤
│                      模型层                              │
│   stdp_model.py (LIF + STDP + 网络构建)                  │
│   model.py      (LIF仿真 + Hebbian更新 + WTA)           │
│   (神经元动力学, 突触可塑性, 侧向抑制, 内在可塑性)         │
├─────────────────────────────────────────────────────────┤
│                      编码层                              │
│   utils.py (MNIST加载, 泊松编码, 可视化)                  │
│   (图像→脉冲序列转换, 数据预处理)                         │
├─────────────────────────────────────────────────────────┤
│                      引擎层                              │
│   Brian2 (Cython C代码生成) | Numba (JIT编译)            │
│   (微分方程求解, 事件驱动仿真, 硬件加速)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 方法一：在线三元STDP (Brian2)

### 原理

严格复现 Diehl & Cook (2015) 的无监督STDP学习框架。

**神经元模型 — 电导型LIF**：

```
τ_m · dv/dt = (v_rest - v) + I_exc + I_inh

I_exc = ge · (-v)          ← 电导型兴奋输入: 驱动电位为 0mV
I_inh = gi · (-100 - v)    ← 电导型抑制输入: 驱动电位为 -100mV

ge → 0  (τ=1ms)            ← 兴奋电导快速衰减
gi → 0  (τ=2ms)            ← 抑制电导稍慢衰减
```

膜时间常数 `τ_m = 100ms`（较长，增强发放率估计稳定性）。

**STDP学习规则 — 三重trace在线更新**：

```
pre trace:       τ=20ms   输入脉冲到达 → pre=1
post1 trace:     τ=20ms   神经元发放 → post1=1
post2 trace:     τ=40ms   神经元发放 → post2=1

LTD (输入脉冲触发时):   Δw = -0.0001 × post1
LTP (神经元发放触发时): Δw = +0.01 × pre × post2_before

权重裁剪: w ∈ [0, 1.0]
每样本前归一化: Σw[j] = 78.0
```

**网络架构**：

```
输入层 (784 泊松神经元)
   │
   │  全连接 (STDP可塑)
   ↓
兴奋层 (1200 LIF神经元)  ←→  抑制层 (1200 抑制性神经元)
   │        E→I: 一对一 (w=10.4)      │
   │        I→E: 全连接除自身 (w=17.0) │
   │                                 │
   └── 涌现式 WTA 竞争 ────────────────┘
```

**仿真流程**（每个训练样本）：

```
1. 权重归一化: Σw[j] = 78.0
2. 350ms 刺激期: 泊松脉冲输入 + 在线STDP权重更新
3. 150ms 静息期: 零输入, trace衰减回零
4. 输入强度自适应: 总发放 < 5 → intensity+1 → 重新展示
```

**标签分配**（observe阶段）：

```
1. 冻结权重, 冻结theta
2. 对每个类别: 收集所有训练样本在该类别的神经发放模式
3. 计算每个神经元的类别平均响应
4. 分配: assign[神经元] = argmax(各类别平均响应)
```

**推理**（test阶段）：

```
1. 展示测试样本, 得到各神经元发放次数
2. 每个神经元投给其分配的标签, 票数 = 发放次数
3. 得票最多的类别 = 预测结果
```

### 依赖

| 包 | 作用 |
|------|------|
| Brian2 | 脉冲神经网络仿真引擎，Cython代码生成加速 |
| NumPy | 矩阵运算、数据存储 |
| Matplotlib | 感受野和神经元分配可视化 |
| progressbar2 | 训练进度条 |

---

## 方法二：STDP速率近似 / Hebbian竞争学习 (Numba)

### 原理

基于 Zenke et al. (2015) 的STDP稳态平衡理论：**当LTP和LTD达到稳态时，权重向量方向收敛到输入模式聚类中心**。

这意味着在稳态下：
- STDP的LTP ⟺ Hebbian LTP（赢家权重向输入模式靠拢）
- STDP的LTD ⟺ 权重归一化（全局约束，保持总权重恒定）

因此可以用更简单的速率近似替代在线STDP。

**学习规则**（每个样本，仿真后一次性更新）：

```
1. 泊松编码 + LIF仿真 (50ms)
2. 选择赢家: 发放次数最多的神经元（排除不应期内的）
3. Hebbian LTP: w[赢家] += lr × image
4. 权重裁剪: clamp(w, 0, w_max)
5. 权重归一化: w[赢家] *= target / sum(w[赢家])
6. 不应期轮换: 赢家进入50样本的不应期
```

**关键区别 vs 在线STDP**：

| 维度 | 在线STDP | 速率近似 |
|------|----------|----------|
| 权重更新时机 | 仿真中每个spike触发 | 仿真结束后一次性完成 |
| 对spike timing的依赖 | 严格依赖（三重trace记录时序） | 不依赖（仅依赖发放率） |
| 竞争方式 | E↔I回路涌现式WTA | 不应期轮换 + 显式argmax |
| 神经元数 | 1200 | 3000 |
| 仿真时长 | 350ms | 50ms |
| 训练速度 | ~12.5小时 | 17分钟 |
| 框架 | Brian2 (Cython C代码) | Numba (JIT编译) |

### 额外机制：不应期轮换

Hebbian版本使用了STDP版本没有的显式不应期轮换机制：

```
每个神经元赢了一次后 → 进入50样本不应期 → 无法再次获胜
→ 强制其他神经元参与竞争 → 防止垄断
```

这是对STDP + 内在可塑性中"自平衡"效应的显式模拟。在STDP版本中，自适应theta + 权重归一化自然地实现了类似的负载均衡。

### 依赖

| 包 | 作用 |
|------|------|
| NumPy | 矩阵运算、数据加载/存储 |
| Numba | JIT编译加速LIF仿真（多核并行） |
| Matplotlib | 感受野和神经元分配可视化 |
| tqdm | 训练进度条 |

---

## 训练集与测试集分割

| 用途 | 样本数 | 说明 |
|------|------|------|
| **训练集** | 60,000 | MNIST官方训练集，无监督STDP/Hebbian学习（标签不参与权重更新） |
| **测试集** | 10,000 | MNIST官方测试集，**仅用于最终评估，不参与训练或调参** |
| **标签分配集** | 前N个训练样本 | 用于事后标签分配（observe阶段），权重保持冻结 |

---

## 生成图表说明

### receptive_fields — 感受野

每个神经元的784维权重reshape为28×28的灰度图。

- **清晰数字** → 神经元高度特化于识别该类数字
- **模糊模板** → 多类别混合激活，特征被平均化
- **稀疏特征** (STDP版) → STDP仅强化时序相关的输入通道，特征更局部化
- **平滑特征** (Hebbian版) → 速率近似学习到更完整的数字模板

对比两种方法的感受野可以直观看到STDP的时序敏感性对特征学习的影响。

### neuron_assignment — 神经元分配

- **左图**：各类别分配的神经元数量 — 反映竞争学习的资源分配是否均衡
- **右图**：各类别最强神经元的平均发放数 — 反映特化程度，柱子越高说明至少存在对该类别高度特化的专家神经元

---


## 运行指南



| 方法 | 训练+测试 | 只测试 |
|------|-----------|--------|
| **STDP** (Brian2) | `cd stdp_online && python train_stdp.py` | `cd stdp_online && python test_brian.py` |
| **Hebbian** (Numba) | `cd stdp_hebbian && python main.py` | `cd stdp_hebbian && python main.py --test` |

- `train_stdp.py`：自动完成 **训练 → 标签分配 → 测试** 全流程，约12.5小时
- `main.py`（无参数）：自动完成 **训练 → 标签分配 → 测试 → 保存模型**，约17分钟
- `test_brian.py`：加载预训练模型，秒出92.39%准确率
- `main.py --test`：跳过训练，加载 `model_hebbian.npz` 直接测试，约1分钟

> Hebbian 也可以用 `cd stdp_hebbian && python test.py` 只测试（推理+可视化），但不会更新混淆矩阵；`main.py --test` 会重新生成 `confusion_hebbian.npy`。

---

### 方法一：在线三元STDP (Brian2)

#### 1. 推理（使用预训练模型）

```bash
cd stdp_online
python test_brian.py
```

输出：
- **实时推理**：对100个样本逐张预测，展示在线推理速度
- **完整准确率**：从预存的 `confusion.npy` 直接读取10K测试集结果（**92.39%**）
- **可视化**：生成 `receptive_fields_stdp.png`（感受野）和 `neuron_assignment_stdp.png`（神经元分配）

预训练模型位置：`stdp_online/data/stdp_full/`（`weights.npy` + `assign.npy` + `confusion.npy`）

#### 2. 完整训练（约12.5小时）

```bash
cd stdp_online
python train_stdp.py
```

训练流程（三阶段，全自动）：

| 阶段 | 函数 | 样本数 | 耗时 | 输出 |
|------|------|------|------|------|
| **训练** | `train()` | 180K（60K×3轮） | ~12.5h | 在线STDP权重更新 |
| **标签分配** | `observe()` | 5K | ~10min | 神经元→类别映射表 `assign.npy` |
| **测试** | `test()` | 10K | ~15min | 混淆矩阵 `confusion.npy` |

训练输出目录 `stdp_online/data/stdp_full/`：

| 文件 | 大小 | 内容 |
|------|------|------|
| `weights.npy` | 7.2 MB | 输入→兴奋 STDP权重 (784×1200) |
| `assign.npy` | 9.5 KB | 神经元→类别分配表 (1200个整数) |
| `theta.npy` | 9.5 KB | 最终自适应阈值 |
| `confusion.npy` | 928 B | 10×10混淆矩阵 |
| `train_stats.npy` | 9.6 MB | 训练过程统计序列 |
| `train_w_hist.npy` | 725 MB | 权重历史快照 |
| `observe_stats.npy` | 274 KB | 标签分配阶段统计 |



#### 3. 关键超参数

| 参数 | 值 | 说明 |
|------|------|------|
| `N_NEURONS` | 1200 | 兴奋神经元数 |
| `N_TRAIN` | 180000 | 训练样本数（60K×3轮） |
| `N_OBSERVE` | 5000 | 标签分配样本数 |
| `N_TEST` | 10000 | 测试样本数 |
| `SEED` | 42 | 固定随机种子 |

修改超参数：编辑 `train_stdp.py` 中的 `dc.N_NEURONS`、`dc.N_TRAIN` 等变量。

---

### 方法二：STDP速率近似 / Hebbian (Numba)

#### 1. 推理（使用预训练模型）

```bash
cd stdp_hebbian
python test.py
```

输出：
- **实时推理**：对100个样本逐张预测
- **完整准确率**：从预存的 `confusion_hebbian.npy` 读取10K测试集结果（**91.90%**）
- **可视化**：生成 `receptive_fields_hebbian.png` 和 `neuron_assignment_hebbian.png`

预训练模型：`stdp_hebbian/model_hebbian.npz`

#### 2. 完整训练（约17分钟）

```bash
cd stdp_hebbian
python main.py
```


训练流程：

| 阶段 | 样本数 | 耗时 | 说明 |
|------|------|------|------|
| **初始化** | 采样 | <1s | 每类均匀采样300个样本初始化权重 |
| **训练（4轮）** | 240K | ~15min | Hebbian LTP + 权重归一化 |
| **标签分配** | 30K | ~1min | 统计赢家归属 |
| **测试** | 10K | ~1min | Top-K投票 |

输出文件：

| 文件 | 大小 | 内容 |
|------|------|------|
| `model_hebbian.npz` | 18 MB | 完整训练模型（权重+阈值+标签分配） |
| `confusion_hebbian.npy` | 928 B | 10×10混淆矩阵 |
| `receptive_fields_hebbian.png` | - | 感受野可视化 |
| `neuron_assignment_hebbian.png` | - | 神经元分配图 |

#### 3. 仅测试（不训练）

如果已有 `model_hebbian.npz`，跳过训练直接测试：

```bash
cd stdp_hebbian
python main.py --test
```

也可以指定模型路径：

```bash
python main.py --test --model-path /path/to/your_model.npz
```

#### 4. 关键超参数

| 参数 | 值 | 说明 |
|------|------|------|
| `N_EXCITATORY` | 3000 | 兴奋神经元数（每类~300个） |
| `DURATION_MS` | 50 | 每样本仿真时长(ms) |
| `N_EPOCHS` | 4 | 训练轮数 |
| `LR` | 0.02 | Hebbian学习率 |
| `REF_PERIOD_SAMPLES` | 50 | 赢家不应期(样本数) |
| `MAX_RATE_HZ` | 400 | 泊松编码最大发放率(Hz) |
| `TOP_K_VOTE` | 10 | 推理投票神经元数 |

修改超参数：编辑 `main.py` 中的对应变量。

---

### 快速验证：两方法对比

```bash
# 终端1: STDP推理
cd stdp_online && python test_brian.py

# 终端2: Hebbian推理
cd stdp_hebbian && python test.py
```

两者都会输出各自的准确率和每类精度，可以直接对比。

**预期结果**：

```
方法一 (STDP):   92.39%  |  训练: ~12.5小时  |  模型: 7.2 MB
方法二 (Hebbian): 91.90%  |  训练: ~17分钟    |  模型: 18 MB



## 总结

本项目通过两种互补的方法验证了STDP在SNN手写数字识别中的有效性：

1. **在线三元STDP**（Brian2，严格复现Diehl & Cook 2015）：1200个LIF神经元通过E→I→E抑制回路实现涌现式WTA竞争，在线三重trace STDP在350ms泊松脉冲窗口内实时更新权重，达到 **92.39%** 准确率。

2. **STDP速率近似**（Numba，基于Zenke et al. 2015稳态理论）：将LTP/LTD平衡等价为Hebbian LTP + 权重归一化，3000神经元通过不应期轮换实现均衡竞争，仅 **17分钟** 训练即达到 **91.90%** 准确率。

两种方法共同验证了三个核心结论：
- STDP无监督学习可在SNN中实现高性能模式识别（>90%准确率）
- STDP稳态平衡可被速率近似有效捕获（两种方法准确率仅差0.5%）
- 速率近似为实际应用提供了100倍加速的训练方案，同时几乎不损失精度

---

## 参考

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
- Zenke, F., Agnes, E. J., & Gerstner, W. (2015). Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. *Nature Communications*, 6, 6922.

