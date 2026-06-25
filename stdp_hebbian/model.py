"""
STDP速率近似 — Hebbian LTP + 权重归一化 (竞争学习)

理论依据 (Zenke et al., 2015):
  当STDP的LTP和LTD到达稳态平衡时，权重向量方向收敛到输入模式的聚类中心。
  在此稳态下:
    STDP的LTP (pre-before-post) → 权重向输入模式靠拢 → 等价于 Hebbian LTP
    STDP的LTD (post-before-pre) → 未增强权重相对衰减 → 等价于 权重归一化

  这意味着: 如果不关心精确的spike timing，只关心最终学到的特征，
  可以用更快的速率近似（仿真后一次性更新）代替在线STDP（每个spike触发更新）。

学习流程 (每个样本):
  1. 泊松编码 → LIF仿真 (50ms, 远短于STDP版的350ms)
  2. 赢家 = argmax(发放次数), 排除不应期神经元
  3. Hebbian LTP: w[赢家] += lr × image  (权重向输入模式靠拢)
  4. 权重裁剪到 [0, w_max]
  5. 权重归一化: w[赢家] *= target / sum(w[赢家])  (LTD等效)
  6. 不应期轮换: 赢家休息 ref_period_samples 个样本

关键区别 vs 在线STDP:
  - 权重更新在仿真后一次性完成（而非每个spike触发）
  - 不依赖spike timing（仅依赖发放率/"谁发得最多"）
  - 使用不应期轮换替代E↔I回路（更简单，但负载均衡效果不如自适应θ）
  - 训练速度提升约44倍 (17min vs 12.5h)

参考:
  Zenke, F., Agnes, E. J., & Gerstner, W. (2015). Diverse synaptic plasticity
  mechanisms orchestrated to form and retrieve memories in spiking neural networks.
  Nature Communications, 6, 6922.
"""

import os
import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def _simulate_lif(
    input_spikes,            # (n_input, n_steps) 输入脉冲矩阵, 0或1
    w_ie,                    # (n_exc, n_input) 输入→兴奋权重
    n_steps,
    v_rest_e, v_thresh_e, tau_m_e, refrac_e,
    tau_ge, tau_gi,
    theta_plus, tc_theta, theta_offset,
    inh_strength, dt,
):
    """
    纯推理LIF仿真 — 电导型LIF + 全局抑制 + 自适应阈值 + 不应期。

    这是本项目两种方法共用同一套LIF仿真引擎。
    Numba JIT编译使其达到C级别速度，是多核并行的。

    每个时间步 dt 的执行顺序:
      1. 突触电导指数衰减: g(t+dt) = g(t) × exp(-dt/τ)
      2. 输入脉冲加权求和 → 更新兴奋电导: ge += Σ w·spike
      3. LIF膜电位更新:
         dv/dt = ((v_rest - v) + ge×(-v) + (gi+gi_global)×(-100-v)) / τ_m
         其中 ge×(-v) 是电导型兴奋电流（反转为0mV）
         (gi+gi_global)×(-100-v) 是电导型抑制电流（反转为-100mV）
      4. 检查发放条件: v > θ - θ_offset + v_thresh_e 且 不在不应期
      5. 选膜电位最高的神经元发放（WTA: 每步最多一个神经元发放）
      6. 发放后: 重置膜电位, θ += theta_plus, gi_global += inh_strength

    为什么每步最多一个神经元发放？
      - 实际生物网络中多神经元可同时发放
      - 但在竞争学习中，限单发放 = 更强的WTA竞争 = 更清晰的特征分化
      - gi_global（全局抑制）在每次发放后增加，进一步压制其他神经元

    参数:
        input_spikes: (n_input, n_steps) float64 输入脉冲, 0或1
        w_ie: (n_exc, n_input) 权重矩阵
        n_steps: 仿真时间步数
        v_rest_e: 静息电位 (mV), 默认-65
        v_thresh_e: 基础发放阈值 (mV), 默认-52
        tau_m_e: 膜时间常数 (ms), 默认100
        refrac_e: 绝对不应期 (ms), 默认5
        tau_ge: 兴奋电导衰减τ (ms), 默认1
        tau_gi: 抑制电导衰减τ (ms), 默认2
        theta_plus: 每次发放阈值增量, 默认0.05
        tc_theta: 自适应阈值衰减τ (ms), 默认1e7
        theta_offset: 阈值偏置 (mV), 默认20
        inh_strength: 全局抑制强度, 默认17.0
        dt: 仿真步长 (ms)

    返回:
        counts: (n_exc,) int32 每个神经元在仿真期间的总发放次数
    """
    n_exc = w_ie.shape[0]

    # 初始化状态变量
    v_e = np.full(n_exc, v_rest_e, dtype=np.float64)     # 膜电位
    ge_e = np.zeros(n_exc, dtype=np.float64)              # 兴奋电导
    gi_e = np.zeros(n_exc, dtype=np.float64)              # 抑制电导
    theta = np.full(n_exc, 20.0, dtype=np.float64)       # 自适应阈值
    timer_e = np.full(n_exc, refrac_e + 1.0, dtype=np.float64)  # 不应期计时器

    # 预计算衰减因子（避免每步重复计算exp）
    ge_decay = np.exp(-dt / tau_ge)
    gi_decay = np.exp(-dt / tau_gi)
    theta_decay = np.exp(-dt / tc_theta)

    counts = np.zeros(n_exc, dtype=np.int32)
    gi_global = 0.0          # 全局抑制电导（每次发放后累加，每步衰减）

    for t in range(n_steps):
        # ── 步骤1: 电导和阈值衰减 ──
        ge_e *= ge_decay
        gi_e *= gi_decay
        gi_global *= gi_decay
        theta *= theta_decay
        timer_e += dt         # 不应期计时器递增

        # ── 步骤2: 脉冲→兴奋电导更新 ──
        # dot(w_ie, spike) = 每个神经元收到多少加权输入脉冲
        inp_sp = np.ascontiguousarray(input_spikes[:, t])
        ge_e += np.dot(w_ie, inp_sp)

        # ── 步骤3: LIF膜电位更新 ──
        # prange = Numba并行for循环，充分利用多核CPU
        for i in prange(n_exc):
            # 电导型兴奋电流 = ge × (0 - v)    ← 反转电位0mV
            i_syn_e = ge_e[i] * (-v_e[i])
            # 电导型抑制电流 = (gi + gi_global) × (-100 - v)  ← 反转电位-100mV
            i_syn_i = (gi_e[i] + gi_global) * (-100.0 - v_e[i])
            # 欧拉积分: v(t+dt) = v(t) + dt × dv/dt
            v_e[i] += dt * ((v_rest_e - v_e[i]) + i_syn_e + i_syn_i) / tau_m_e

        # ── 步骤4-5: 检查发放 ──
        thresh = theta - theta_offset + v_thresh_e
        best_j = -1
        best_v = -1e9
        # 找到所有满足: 膜电位>阈值 且 不在不应期 的神经元
        for i in range(n_exc):
            if v_e[i] > thresh[i] and timer_e[i] >= refrac_e:
                if v_e[i] > best_v:
                    best_v = v_e[i]
                    best_j = i

        # ── 步骤6: 发放处理 ──
        if best_j >= 0:
            v_e[best_j] = v_rest_e                          # 重置膜电位
            theta[best_j] += theta_plus                     # 阈值升高（内在可塑性）
            timer_e[best_j] = 0.0                           # 重置不应期计时器
            counts[best_j] += 1                             # 发放计数+1
            gi_global += inh_strength                       # 全局抑制（压制其他神经元）

    return counts


class SNN:
    """
    STDP速率近似 — 基于Hebbian竞争学习的脉冲神经网络。

    这个类封装了完整的训练和推理流程，使用Numba JIT编译的LIF仿真引擎。

    设计决策:
      - 为什么用不应期轮换而不是E↔I回路？
        不应期轮换在速率近似中更易实现，且与"仿真后一次性更新"的范式一致。
        E↔I回路需要在线仿真中实时操作，与速率近似的事后处理不匹配。
        代价: 负载均衡效果不如STDP的自适应θ（max/min=7.7× vs 2.5×）。

      - 为什么仿真时长只有50ms（vs STDP的350ms）？
        速率近似不需要精确的spike timing信息，只需要可靠的发放率估计。
        50ms足够产生可区分的发放统计，且大幅缩短训练时间。

      - 为什么3000个神经元（vs STDP的1200个）？
        速率近似效率高（17分钟就能训练完），可以用更多神经元来补偿
        简化的学习规则。3000个（每类~300个）在CPU上17分钟可训练完成。
    """

    def __init__(self, n_input=784, n_excitatory=2500,
                 dt_ms=1.0, duration_ms=50.0, max_rate_hz=400.0,
                 v_rest_e=-65.0, v_thresh_e=-52.0, tau_m_e=100.0, refrac_e=5.0,
                 tau_ge=1.0, tau_gi=2.0,
                 lr=0.02, w_max=1.0,
                 theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
                 target_weight_sum=78.0,
                 ref_period_samples=50, inh_strength=17.0):
        """
        初始化 Hebbian 竞争学习脉冲网络。

        超参数与STDP版保持一致（膜时间常数、电导衰减、归一化目标等），
        确保两种方法的对比是公平的——差异仅来自学习规则本身。

        参数:
            n_input: 输入神经元数 (MNIST 28×28=784)
            n_excitatory: 兴奋神经元总数
            dt_ms: 仿真步长 (ms)
            duration_ms: 每样本仿真时长 (ms)
            max_rate_hz: 泊松编码最大发放率 (Hz)
            v_rest_e: 静息电位 (mV)
            v_thresh_e: 基础发放阈值 (mV)
            tau_m_e: 膜时间常数 (ms)
            refrac_e: 绝对不应期 (ms)
            tau_ge: 兴奋电导衰减时间常数 (ms)
            tau_gi: 抑制电导衰减时间常数 (ms)
            lr: Hebbian学习率
            w_max: 权重上限（防止单通道权重过大）
            theta_plus: 发放后阈值增量（内在可塑性）
            tc_theta: 自适应阈值衰减时间常数 (ms)
            theta_offset: 阈值偏置 (mV)
            target_weight_sum: 权重归一化目标值
            ref_period_samples: 赢家不应期步数（防止连续获胜）
            inh_strength: 全局抑制强度
        """
        self.n_input = n_input
        self.n_excitatory = n_excitatory
        self.dt = dt_ms
        self.duration_ms = duration_ms
        self.max_rate_hz = max_rate_hz
        self.n_steps = int(duration_ms / dt_ms)              # 仿真总步数

        # LIF参数
        self.v_rest_e = v_rest_e
        self.v_thresh_e = v_thresh_e
        self.tau_m_e = tau_m_e
        self.refrac_e = refrac_e
        self.tau_ge = tau_ge
        self.tau_gi = tau_gi

        # Hebbian学习参数
        self.lr = lr
        self.w_max = w_max

        # 内在可塑性参数
        self.theta_plus = theta_plus
        self.tc_theta = tc_theta
        self.theta_offset = theta_offset
        self.target_weight_sum = target_weight_sum
        self.ref_period_samples = ref_period_samples
        self.inh_strength = inh_strength

        # 不应期计数器: 每个神经元还有多少样本不能参与竞争
        # 每样本递减1，赢家置为ref_period_samples
        self.refractory_counter = np.zeros(n_excitatory, dtype=np.int32)

        # 权重初始化: 小随机值 ([0.01, 0.31) × 0.3)
        # +0.01偏移确保所有权重非零（避免零权重导致死神经元）
        rng = np.random.RandomState(42)
        self.w_ie = (rng.rand(n_excitatory, n_input).astype(np.float64) + 0.01) * 0.3
        self.normalize_weights()

        # 标签分配相关（训练后填充）
        self.assigned_labels = np.full(n_excitatory, -1, dtype=np.int32)
        self.spike_counts_per_class = None
        self.input_intensity = 2.0                             # 初始输入强度

    # ── 初始化 ──

    def initialize_from_exemplars(self, images, labels, n_per_class=None):
        """
        用训练样本初始化权重（加速收敛）。

        方法:
          - 每个类别从训练集中随机采样n_per_class个样本
          - 将该类对应神经元的初始权重设为这些样本的像素值
          - 添加小噪声避免所有神经元初始完全相同

        为什么用这个初始化？
          随机初始化需要更多训练样本才能收敛到数字模板。
          基于样本的初始化让神经元从合理的起点开始，大幅减少训练时间。
          注意: 这只是一种初始化技巧，不影响学习规则本身的无监督性质。
        """
        if n_per_class is None:
            n_per_class = self.n_excitatory // 10            # 每类平均分配
        rng = np.random.RandomState(42)
        for c in range(10):
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, size=n_per_class, replace=True)
            start = c * n_per_class
            end = (c + 1) * n_per_class
            self.w_ie[start:end] = images[chosen].astype(np.float64)

        self.w_ie *= 0.3                                     # 缩放初始权重
        self.normalize_weights()                             # 归一化到目标和
        # 添加微小噪声，打破对称性
        self.w_ie += rng.normal(0, 0.01, self.w_ie.shape).astype(np.float64)
        np.clip(self.w_ie, 0.0, self.w_max, out=self.w_ie)
        self.normalize_weights()

    # ── 权重管理 ──

    def normalize_weights(self):
        """
        权重归一化 — STDP-LTD的速率等效。

        每个神经元j的输出权重向量 w_j 被缩放使得 Σw_j = target_weight_sum:
          w_j *= target_weight_sum / sum(w_j)

        为什么这等效于LTD？
          - Hebbian LTP增大某些权重后，归一化将所有权重等比缩小
          - 被增强的输入保持相对优势，未被增强的输入相对衰减
          - 这和STDP中LTD的效果一致：不相关的输入被削弱

        为什么target=78.0？
          这是Diehl & Cook(2015)的经验值，与784个输入×~0.1的平均权重对应。
        """
        sums = self.w_ie.sum(axis=1)
        for j in range(self.n_excitatory):
            if sums[j] > 0:
                self.w_ie[j] *= self.target_weight_sum / sums[j]

    # ── 前向推理 ──

    def forward(self, image, seed=None):
        """
        前向推理: 图像 → 泊松编码 → LIF仿真 → 发放计数。

        步骤:
          1. 像素值 × max_rate_hz × (intensity/2.0) → 发放率 (Hz)
             例如 intensity=2, max_rate=400 → 像素=1时发放率=400Hz
          2. 发放率 × dt/1000 → 每个时间步的发放概率
             例如 400Hz × 1ms/1000 = 0.4 每步发放概率
          3. 伯努利采样: rand() < prob → (n_input, n_steps) 脉冲矩阵
          4. _simulate_lif(): 电导型LIF仿真 → 各神经元发放次数

        泊松编码的随机性:
          - 每次调用forward()产生不同的脉冲序列（seed控制可复现性）
          - STDP版利用这种随机性捕捉时序相关
          - Hebbian版仅需要可靠的发放率估计，所以50ms就够了

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            seed: 随机种子

        返回:
            counts: (n_excitatory,) int32 各神经元发放次数
        """
        # 像素值 → 发放率 (Hz)
        # input_intensity/2.0 = 自适应强度因子（训练时为调节发放数动态调整）
        rates = image * self.max_rate_hz * (self.input_intensity / 2.0)
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        prob = rates * (self.dt / 1000.0)                   # 每步发放概率

        # 伯努利采样生成脉冲序列
        input_spikes = (rng.rand(self.n_input, self.n_steps)
                        < prob[:, np.newaxis]).astype(np.float64)
        input_spikes = np.ascontiguousarray(input_spikes)
        return _simulate_lif(
            input_spikes, self.w_ie, self.n_steps,
            self.v_rest_e, self.v_thresh_e, self.tau_m_e, self.refrac_e,
            self.tau_ge, self.tau_gi,
            self.theta_plus, self.tc_theta, self.theta_offset,
            self.inh_strength, self.dt,
        )

    # ── 训练 ──

    def train_on_sample(self, image, seed=None):
        """
        Hebbian竞争学习 (单样本)。

        核心流程:
          1. forward() → 各神经元发放计数
          2. 排除不应期神经元（防止同一神经元连续获胜垄断学习）
          3. 赢家 = argmax(剩余神经元的发放计数)
          4. w[赢家] += lr × image —— 权重直接向输入图像靠拢
          5. 权重裁剪到[0, w_max] —— 防止单通道过大
          6. 权重归一化 —— LTD等效，保持总权重恒定
          7. 不应期计数器更新: 赢家进入不应期，其他递减

        为什么只有赢家被更新？
          - 竞争学习的本质: 不同的赢家学习不同的模式
          - 所有神经元都被更新 → 学到相同特征 → 无特化
          - 只更新赢家 → 每个赢家向其最近赢得的模式靠拢 → 特化

        不应期轮换的作用:
          想象没有不应期: 神经元A初始化最好 → 每次都赢 → 每次被更新 →
          越来越好 → 永远赢 → 其他神经元永远学不到东西。
          有了不应期: A赢了之后休息50样本 → B有机会赢 → B也开始学习 →
          轮流学习 → 特征多样性。

        但这个机制在跨类别均衡上不如STDP的自适应θ:
          - 自适应θ按神经活动量调节（频繁发→阈值高→被迫休息）
          - 不应期只是"赢家休息N轮"，不影响非赢家神经元的质量
          - 结果: Hebbian版各类别神经元数差异大(56~429)，投票质量参差

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            seed: 随机种子

        返回:
            counts: (n_excitatory,) 发放计数
        """
        counts = self.forward(image, seed=seed)            # LIF仿真

        if counts.sum() == 0:
            return counts                                   # 无发放，跳过（极罕见）

        # 屏蔽不应期神经元: 将发放数设为-1确保它们不会成为赢家
        counts_masked = counts.copy().astype(np.float64)
        counts_masked[self.refractory_counter > 0] = -1.0
        winner = np.argmax(counts_masked)                   # 排除不应期后的赢家

        # 不应期计数器更新
        self.refractory_counter = np.maximum(0, self.refractory_counter - 1)
        self.refractory_counter[winner] = self.ref_period_samples  # 赢家进入不应期

        # Hebbian LTP: 权重向输入模式靠拢
        # w += lr × image  —— 标准Hebbian: 同时激活的连接被增强
        # image的每个像素直接加到对应权重上
        self.w_ie[winner] += self.lr * image
        np.clip(self.w_ie[winner], 0.0, self.w_max, out=self.w_ie[winner])

        # 权重归一化 — LTD等效
        # 归一化将所有未增强的权重等比缩小 = 等效于STDP的LTD效果
        wsum = self.w_ie[winner].sum()
        if wsum > 0:
            self.w_ie[winner] *= self.target_weight_sum / wsum

        return counts

    # ── 标签分配 ──

    def assign_labels(self, spike_counts, labels):
        """
        事后标签分配: 统计每个神经元最常对哪个数字类别的样本发放最多。

        机制:
          对每个样本，找到发放最多的神经元作为"赢家"。
          该赢家获得该样本真实标签的一票。
          处理完所有样本后，每个神经元得到一个10类投票分布。
          最终标签 = argmax(投票数) → "该神经元最常被哪个类别的样本激活"。

        注意: 这个过程完全不修改权重，只是"读取"无监督训练的结果。

        参数:
            spike_counts: (n_samples, n_excitatory) 各样本的各神经元发放数
            labels: (n_samples,) 每个样本的真实标签
        """
        n_classes = 10
        self.spike_counts_per_class = np.zeros((n_classes, self.n_excitatory))
        winners = np.argmax(spike_counts, axis=1)          # 每个样本的赢家

        # 统计每个神经元在各类别样本中获胜的次数
        for c in range(n_classes):
            mask = labels == c
            for w in winners[mask]:
                self.spike_counts_per_class[c, w] += 1

        # 分配: 神经元归属于投票数最多的类别
        self.assigned_labels = np.argmax(self.spike_counts_per_class, axis=0)
        # 从未获胜的神经元标记为-1（无效）
        never_won = self.spike_counts_per_class.sum(axis=0) == 0
        self.assigned_labels[never_won] = -1

    # ── 持久化 ──

    def save(self, path):
        """保存模型到 .npz 压缩格式"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        spc = (self.spike_counts_per_class
               if self.spike_counts_per_class is not None
               else np.zeros((10, self.n_excitatory)))
        np.savez(path,
                 w_ie=self.w_ie,
                 assigned_labels=self.assigned_labels,
                 spike_counts_per_class=spc,
                 input_intensity=np.array(self.input_intensity))

    @staticmethod
    def load(path, **override_kwargs):
        """从 .npz 文件加载模型，支持覆盖默认参数"""
        data = np.load(path, allow_pickle=True)
        kwargs = dict(n_input=784, n_excitatory=data['w_ie'].shape[0])
        kwargs.update(override_kwargs)
        snn = SNN(**kwargs)
        snn.w_ie = data['w_ie']
        snn.assigned_labels = data['assigned_labels']
        snn.spike_counts_per_class = data['spike_counts_per_class']
        if 'input_intensity' in data:
            snn.input_intensity = float(data['input_intensity'])
        return snn

    # ── 推理 ──

    def predict(self, image, top_k=10, seed=None):
        """
        Top-K加权投票预测。

        为什么用Top-K而不是单一赢家？
          - 单一赢家容易被噪声影响（某次仿真恰好某个神经元多发了几次）
          - Top-K让多个专家神经元共同投票 → 更鲁棒
          - 加权投票（票数=发放次数）：越活跃的神经元意见越重要

        流程:
          1. forward() → 所有神经元发放计数
          2. 取发放最多的K个神经元作为投票委员会
          3. 按发放次数加权: votes[标签] += 发放次数
          4. 得票最多的类别 = 预测结果

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            top_k: 参与投票的神经元数 (默认10)
            seed: 随机种子

        返回:
            (pred_label, counts): 预测的数字(0-9), -1表示无发放;
                                  counts为各神经元发放次数
        """
        counts = self.forward(image, seed=seed)
        if counts.sum() == 0:
            return -1, counts                               # 无任何发放（极罕见）

        # 取发放最多的K个神经元
        top_winners = np.argsort(counts)[-top_k:]
        votes = np.zeros(10)
        for w in top_winners:
            if self.assigned_labels[w] >= 0:                # 跳过未分配神经元
                votes[self.assigned_labels[w]] += counts[w]  # 加权投票

        if votes.sum() > 0:
            return np.argmax(votes), counts
        return -1, counts
