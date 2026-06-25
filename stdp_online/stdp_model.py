from brian2 import *
from collections import defaultdict
from pathlib import Path
from progressbar import ProgressBar
from random import randrange, seed as rseed
from struct import unpack
import numpy as np
import itertools
import gzip
import urllib.request
import os

# ── MNIST 下载 URL ──
_MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

# 超参数
MODE = 'test'           # 运行模式: 'train' | 'observe' | 'test' | 'plot'
N_TRAIN = 25_000        # 训练样本总数（循环使用MNIST的60K训练集）
N_OBSERVE = 2_000       # 标签分配阶段使用的样本数
N_TEST = 1_000          # 测试样本数
SEED = 42               # 全局随机种子，确保结果可复现
MNIST_PATH = Path(__file__).resolve().parent.parent / 'data'  # 默认: 项目根/data/
DATA_PATH = Path(__file__).resolve().parent / 'data' / 'stdp_full'  # 默认输出目录

# 网络常数（来自Diehl & Cook 2015论文）
N_INP = 784             # 输入神经元数 = 28×28 MNIST像素
N_NEURONS = 400         # 兴奋神经元数（>1200个更好，400个是最小可运行配置）
V_EXC_REST = -65 * mV   # 兴奋神经元静息电位
V_INH_REST = -60 * mV   # 抑制神经元静息电位
INTENSITY = 2           # 初始输入强度因子（控制泊松编码的发放率缩放）
W_EXC_INH = 10.4        # E→I 一对一连接权重
W_INH_EXC = 17.0        # I→E 全连接（除自身）权重，即抑制强度


# 辅助函数

def save_npy(arr, path):
    """保存 numpy 数组到 .npy 文件"""
    arr = np.array(arr)
    print('%-9s %-15s => %-30s' % ('Saving', arr.shape, path))
    np.save(path, arr)


def load_npy(path):
    """加载 .npy 文件, 返回 numpy 数组"""
    arr = np.load(path)
    print('%-9s %-30s => %-15s' % ('Loading', path, arr.shape))
    return arr


def _ensure_mnist_data():
    """若 MNIST_PATH 下缺少数据文件，自动下载 .gz 压缩包。"""
    MNIST_PATH.mkdir(parents=True, exist_ok=True)
    for url in _MNIST_URLS.values():
        fname = url.split('/')[-1]
        fpath = MNIST_PATH / fname
        if not fpath.exists():
            print(f'  下载 {fname} ...')
            urllib.request.urlretrieve(url, fpath)
            print(f'  完成: {fname}')


def _open_mnist_file(base_name):
    """
    优先打开解压后的文件；若不存在则回退到 .gz 压缩包（透明 gzip 解压）。
    返回 (file_object, 是否为gzip)。
    """
    raw_path = MNIST_PATH / base_name
    gz_path = MNIST_PATH / (base_name + '.gz')
    if raw_path.exists():
        return open(raw_path, 'rb'), False
    if gz_path.exists():
        return gzip.open(gz_path, 'rb'), True
    raise FileNotFoundError(f'MNIST 数据文件不存在: {raw_path} 或 {gz_path}')


def read_mnist(training):
    """
    读取原始MNIST文件（IDX格式，支持 .gz 压缩），返回 (图像, 标签)。

    MNIST IDX文件格式:
      [magic(4B)] [n_images(4B)] [rows(4B)] [cols(4B)] [pixel_data...]
      所有整数均为大端序（big-endian）。

    图像像素值 ÷ 8（归一化到[0, 32]左右），用于后续泊松编码的发放率计算。
    """
    _ensure_mnist_data()                         # 自动下载缺失数据
    tag = 'train' if training else 't10k'

    # 读取图像文件（自动处理 .gz）
    img_file, _ = _open_mnist_file(f'{tag}-images-idx3-ubyte')
    img_file.read(4)                             # 跳过magic number
    n_images = unpack('>I', img_file.read(4))[0] # 图像数量
    n_rows = unpack('>I', img_file.read(4))[0]   # 行数(=28)
    n_cols = unpack('>I', img_file.read(4))[0]   # 列数(=28)

    # 读取标签文件（自动处理 .gz）
    lbl_file, _ = _open_mnist_file(f'{tag}-labels-idx1-ubyte')
    lbl_file.read(8)                             # 跳过magic number + count

    x = np.frombuffer(img_file.read(), dtype=np.uint8)
    x = x.reshape(n_images, -1) / 8.0            # 像素值÷8 → 用于泊松速率
    y = np.frombuffer(lbl_file.read(), dtype=np.uint8)
    img_file.close()
    lbl_file.close()
    return x, y

# 网络构建 — 核心部分

def build_network(training):
    """
    构建完整的STDP脉冲神经网络。

    网络拓扑:
      输入层(784 PoissonGroup)
          ↓ 全连接 STDP可塑突触 (随机延时0-10ms)
      兴奋层(1200 LIF)
          ↕ E↔I回路
      抑制层(1200 抑制性神经元)
          ↓
      SpikeMonitor → 记录发放

    返回:
      Brian2 Network对象
    """
    # ── 基础LIF方程（兴奋层和抑制层共用） ──
    # dv/dt: 膜电位变化 = 漏电流 + 兴奋输入 + 抑制输入
    # i_exc: 电导型兴奋电流 = ge × (0 - v)   —— 驱动电位为0mV
    # i_inh: 电导型抑制电流 = gi × (-100 - v) —— 驱动电位为-100mV
    # dge/dt, dgi/dt: 突触电导以指数衰减（τ=1ms和2ms）
    # dtimer: 距上次发放的时间（用于强制50ms最小发放间隔）
    eqs = '''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    '''
    reset = 'v = %r; timer = 0 * ms' % V_EXC_REST   # 发放后重置

    # ── 兴奋层 ──
    if training:
        # 训练时: 自适应阈值θ动态变化
        # dtheta/dt = -θ/(10^7 ms): 阈值缓慢衰减回基线(20mV)
        # 每次发放后: θ += 0.05mV（阈值升高，更难再次发放）
        # 这个机制称为"内在可塑性"——实现自然负载均衡
        exc_eqs = eqs + '''
        dtheta/dt = -theta / (1e7 * ms)         : volt
        '''
        arr_theta = np.ones(N_NEURONS) * 20 * mV    # 初始阈值为20mV
        reset += '; theta += 0.05 * mV'             # 发放后阈值升高
    else:
        # 推理时: θ固定为训练后的值（从文件加载）
        exc_eqs = eqs + '''
        theta                                   : volt
        '''
        arr_theta = load_npy(DATA_PATH / 'theta.npy') * volt

    # 编译兴奋层方程: τ_mem=100ms(长=发射率稳定), v_rest=-65mV, 抑制反转电位=-100mV
    exc_eqs = Equations(exc_eqs, tau_mem=100 * ms, v_rest=V_EXC_REST, v_inh_base=-100 * mV)

    # 创建兴奋层神经元群
    # threshold: 膜电位超过自适应阈值，且距上次发放>50ms
    # refractory: 绝对不应期5ms（发放后强制休息）
    # method='euler': 欧拉法数值求解微分方程
    ng_exc = NeuronGroup(
        N_NEURONS, exc_eqs,
        threshold='v > (theta - 72 * mV) and (timer > 50 * ms)',
        refractory=5 * ms,
        reset=reset,
        method='euler',
        name='exc')
    ng_exc.v = V_EXC_REST      # 初始膜电位 = 静息电位
    ng_exc.theta = arr_theta   # 初始自适应阈值

    # ── 抑制层 ──
    # 抑制神经元特点: 快时间常数(τ_mem=10ms)、低阈值(-40mV)、短不应期(2ms)
    # 这确保抑制神经元一被激活立即发放，快速抑制其他兴奋神经元
    inh_eqs = Equations(eqs, tau_mem=10 * ms, v_rest=V_INH_REST, v_inh_base=-85 * mV)
    ng_inh = NeuronGroup(N_NEURONS, inh_eqs,
                         threshold='v > -40 * mV',
                         refractory=2 * ms,
                         reset='v = -45 * mV',
                         method='euler',
                         name='inh')
    ng_inh.v = V_INH_REST

    # ── E↔I 侧向抑制回路 ──
    #   1. E→I: 一对一连接(j='i')，权重10.4——兴奋神经元发放时激活同序号的抑制神经元
    #   2. I→E: 全连接除自身("i != j")，权重17.0——抑制神经元发放时抑制所有其他兴奋神经元
    # 效果: 第一个发放的兴奋神经元通过抑制回路压制其他竞争者 → 涌现式WTA
    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre='ge_post += %f' % W_EXC_INH)
    syns_exc_inh.connect(j='i')                    # 一对一: E_i → I_i

    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre='gi_post += %f' % W_INH_EXC)
    syns_inh_exc.connect("i != j")                 # 全连接除自身: I_i → 所有E_j (i≠j)

    # ── 输入层（泊松神经元） ──
    # 每个输入神经元独立地按泊松过程发放脉冲
    # rates在show_sample()中动态设置 = sample * intensity (Hz)
    pg_inp = PoissonGroup(N_INP, 0 * Hz, name='inp')

    # ── 输入→兴奋 突触（STDP可塑） ──
    # on_pre: 输入脉冲到达时触发 → (1)更新兴奋电导 (2)执行LTD权重更新
    # on_post: 神经元发放时触发 → (1)记录post2_before (2)执行LTP权重更新
    model = 'w : 1'                                 # 突触权重变量
    on_post = ''
    on_pre = 'ge_post += w'                         # 基础: 脉冲→电导更新
    if training:
        # === 训练模式: 在线三重trace STDP ===
        # LTD (输入脉冲时): 若目标神经元刚发放过(post1>0)，说明是post-before-pre
        #   → 这个输入与发放无关(甚至是竞争性的) → 削弱权重
        #   Δw = -0.0001 × post1（nu_pre=0.0001，LTD学习率小）
        on_pre += '; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)'

        # LTP (发放时): pre>0说明某输入刚在发放前到达 → pre-before-post时序
        #   → 这个输入可能触发了发放 → 增强权重
        #   Δw = +0.01 × pre × post2_before（nu_post=0.01，LTP学习率是LTD的100倍）
        # post2_before: 发放前瞬间的post2值——确保是pre-before-post而非post-before-pre
        # 发放后: post1=1, post2=1（trace置位，之后按指数衰减）
        on_post += 'post2bef = post2; w = clip(w + 0.01 * pre * post2bef, 0, 1.0); post1 = 1.; post2 = 1.'

        # 三重trace变量（均为event-driven，仅在spike时更新，仿真效率高）:
        #   pre:   τ=20ms，输入脉冲到达时置1，记录最近输入历史（LTP用）
        #   post1: τ=20ms，发放时置1，快速衰减——记录最近发放历史（LTD用）
        #   post2: τ=40ms，发放时置1，慢速衰减——提供更长的时间窗口（LTP调制用）
        # τ_post2 > τ_post1: LTD和LTP用不同时间常数，避免同一事件中同时触发两者
        model += '''
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        '''
        # 初始权重: 随机小值([0.01,0.31)×0.3)，加偏移确保非零
        weights = (np.random.random(N_INP * N_NEURONS) + 0.01) * 0.3
    else:
        # 推理模式: 加载已训练权重
        weights = load_npy(DATA_PATH / 'weights.npy')

    # 创建输入→兴奋的全连接突触
    # connect(True): 全连接（所有784→所有N_NEURONS）
    # delay: 随机0-10ms延时——模拟真实神经传导的分散性，避免所有输入同时到达
    syns_inp_exc = Synapses(pg_inp, ng_exc, model=model, on_pre=on_pre, on_post=on_post, name='inp_exc')
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = 'rand() * 10 * ms'
    syns_inp_exc.w = weights

    # ── 组装网络 ──
    # SpikeMonitor: 记录兴奋神经元的发放事件（用于推理和统计）
    exc_mon = SpikeMonitor(ng_exc, name='sp_exc')
    net = Network([pg_inp, ng_exc, ng_inh, syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon])
    net.run(0 * ms)    # 初始化网络状态
    return net


# ═══════════════════════════════════════════════════════════════
# 仿真与推理
# ═══════════════════════════════════════════════════════════════

def show_sample(net, sample, intensity):
    """
    输入单张图像，运行一次完整的脉冲仿真（350ms刺激 + 150ms静息）。

    流程:
      1. 记录当前各神经元的发放计数
      2. 设置输入层发放率 = sample × intensity (Hz)
      3. 运行350ms仿真（刺激期）——泊松脉冲输入 + STDP在线更新
      4. 将输入层发放率设为0Hz
      5. 运行150ms仿真（静息期）——trace和电导衰减归零
      6. 计算本次的发放计数差

    自适应输入强度:
      若总发放数 < 5 → 输入太弱，intensity+1 → 递归重试
      这是Diehl & Cook(2015)的"输入强度自适应"机制，确保每个样本
      都引起足够的神经活动，避免因输入太弱导致学习停滞。

    返回:
      pat: (N_NEURONS,) 各神经元在本次仿真中的发放次数
    """
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]                           # 记录当前发放计数

    net['inp'].rates = sample * intensity * Hz         # 设置输入发放率
    net.run(350 * ms)                                  # 刺激期: 350ms

    next = exc_mon.count[:]                            # 记录发放后计数
    net['inp'].rates = 0 * Hz                          # 关闭输入
    net.run(150 * ms)                                  # 静息期: 150ms

    pat = next - prev                                  # 本次仿真净发放数
    cnt = np.sum(pat)
    if cnt < 5:
        # 总发放数太少 → 提高输入强度 → 递归重试
        return show_sample(net, sample, intensity + 1)
    return pat


def predict(groups, rates):
    """
    根据各组神经元的平均发放率进行预测（用于推理）。

    参数:
      groups: list of arrays — 每个类别对应的神经元索引列表
      rates: (N_NEURONS,) — 各神经元的发放次数

    返回:
      预测类别 (0-9): 组平均发放率最高的类别
    """
    return np.argmax([rates[grp].mean() for grp in groups])


def normalize_plastic_weights(syns):
    """
    权重归一化: 每个神经元的输入权重总和 = 78.0。

    这是STDP-LTD的全局稳态机制:
    - LTP增大权重 → 归一化将所有权重等比缩小
    - 等效于: 被LTP增强的输入保持相对优势，未增强的输入相对衰减
    - 权重方向(特征模式)不变，仅总模长被约束

    每样本训练前执行——确保权重不会因持续LTP而无限增长。
    """
    conns = np.reshape(syns.w, (N_INP, N_NEURONS))     # → (784, N_NEURONS)
    col_sums = np.sum(conns, axis=0)                   # 每列(神经元)的权重和
    conns *= 78. / col_sums                            # 缩放到目标和=78
    syns.w = conns.reshape(-1)                         # 写回synapses


def stats(net):
    """
    收集网络运行状态快照，用于监控训练过程。

    返回列表:
      [仿真时刻, 总发放数, w均值, w标准差, θ均值, θ标准差]

    用途:
      - 监控权重是否收敛
      - 监控自适应θ的分布变化
      - 诊断死神经元（发放=0）或过活跃神经元
    """
    tick = defaultclock.timestep[:]                    # 当前仿真时刻
    cnt = np.sum(net['sp_exc'].count[:])               # 兴奋层总发放数
    inp_exc = net['inp_exc']
    w_mu = np.mean(inp_exc.w)                          # 权重均值
    w_std = np.std(inp_exc.w)                          # 权重标准差
    exc = net['exc']
    theta = exc.theta / mV                             # 阈值(mV)
    theta_mu = np.mean(theta)                          # 阈值均值
    theta_sig = np.std(theta)                          # 阈值标准差
    return [tick, cnt, w_mu, w_std, theta_mu, theta_sig]


# ═══════════════════════════════════════════════════════════════
# 三阶段流水线
# ═══════════════════════════════════════════════════════════════

def train():
    """
    阶段1 - STDP无监督训练。

    循环展示训练样本:
      1. 权重归一化（每样本前执行，防止权重无限增长）
      2. show_sample(): 350ms仿真 + 在线STDP权重更新
      3. 记录训练统计和权重历史快照

    训练过程的权重更新完全依赖局部spike timing，不使用任何标签信息。
    训练结束后保存:
      - weights.npy:        最终STDP权重
      - theta.npy:          自适应阈值
      - train_stats.npy:    训练统计序列
    """
    X, Y = read_mnist(True)                            # 加载训练集
    n_samples = X.shape[0]
    net = build_network(True)                          # 构建训练网络（STDP启用）
    rows = [stats(net) + [-1]]                         # 统计序列（初始状态）

    for i in ProgressBar()(range(N_TRAIN)):
        ix = i % n_samples                              # 循环使用训练集
        normalize_plastic_weights(net['inp_exc'])      # 每样本前归一化
        show_sample(net, X[ix], INTENSITY)             # 仿真 + STDP更新
        rows.append(stats(net) + [Y[ix]])              # 记录统计 + 真实标签

    save_npy(rows, DATA_PATH / 'train_stats.npy')
    save_npy(net['inp_exc'].w, DATA_PATH / 'weights.npy')
    save_npy(net['exc'].theta, DATA_PATH / 'theta.npy')


def observe():
    """
    阶段2 - 标签分配（权重冻结）。

    在STDP无监督训练后，神经元自发形成了对不同数字类别的响应偏好。
    这一步通过统计来"标注"每个神经元代表哪个数字:

      1. 冻结权重和θ（不再更新）
      2. 对每个训练样本做前向推理，记录各神经元的发放
      3. 按类别分组，计算每个神经元对各类别的平均响应
      4. assign[n] = argmax(各类别平均响应)
         ——即神经元n对哪个数字反应最强，就被分配到哪个数字

    这个阶段不需要标签参与学习——它只是"读取"神经元已经形成的偏好。
    保存:
      - assign.npy: 神经元→类别分配表
      - observe_stats.npy: 标签分配阶段的统计
    """
    X, Y = read_mnist(True)
    n_samples = X.shape[0]
    net = build_network(False)                         # 推理模式：权重从文件加载
    rows = [stats(net) + [-1]]
    responses = defaultdict(list)                      # {类别: [各神经元发放向量列表]}

    for i in ProgressBar()(range(N_OBSERVE)):
        ix = i % n_samples
        exc = show_sample(net, X[ix], INTENSITY)       # 前向推理（权重不更新）
        rows.append(stats(net) + [Y[ix]])
        responses[Y[ix]].append(exc)                   # 按真实标签收集发放模式

    # 计算每个神经元对各类别的平均发放数
    res = np.zeros((10, N_NEURONS))                    # (10, N_NEURONS)
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)         # 类别cls的发放均值

    # 分配: 每个神经元归属到平均发放最强的类别
    assign = np.argmax(res, axis=0)
    save_npy(assign, DATA_PATH / 'assign.npy')
    save_npy(rows, DATA_PATH / 'observe_stats.npy')


def test():
    """
    阶段3 - 测试评估。

    用测试集评估分类准确率:
      1. 加载训练好的权重和标签分配
      2. 对每个测试样本: 前向推理 → 每组神经元平均发放率 → 选最高组
      3. 统计混淆矩阵

    Top-K投票机制（通过predict函数实现）:
      - 将神经元按分配的标签分组
      - 每组计算组内平均发放次数
      - 发放率最高的组 = 预测类别

    保存:
      - confusion.npy: 10×10混淆矩阵（归一化后每行和=1）
    """
    conf = np.zeros((10, 10))                          # 混淆矩阵
    assign = np.load(DATA_PATH / 'assign.npy')         # 加载标签分配
    groups = [np.where(assign == i)[0] for i in range(10)]  # 各类别神经元索引

    X, Y = read_mnist(False)                           # 加载测试集（仅用于评估）
    net = build_network(False)
    for i in ProgressBar()(range(N_TEST)):
        ix = randrange(len(X))                         # 随机抽样测试
        exc = show_sample(net, X[ix], INTENSITY)
        guess = predict(groups, exc)
        conf[Y[ix], guess] += 1

    print('Accuracy: %6.3f' % (np.trace(conf) / np.sum(conf)))
    conf = conf / conf.sum(axis=1)[:, None]            # 归一化: 每行和=1
    print(np.around(conf, 2))
    save_npy(conf, DATA_PATH / 'confusion.npy')


def plot():
    """绘制混淆矩阵热力图，显示各类别的识别准确率"""
    conf = np.load(DATA_PATH / "confusion.npy")
    import matplotlib.pyplot as plt
    plt.imshow(100 * conf, interpolation="nearest", cmap=plt.cm.Blues)
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i, j] == 0:
            continue
        plt.text(j, i, f"{round(100 * conf[i, j])}%",
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if conf[i, j] > 0.5 else "black")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


if __name__ == '__main__':
    seed(SEED)
    rseed(SEED)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    cmds = dict(train=train, observe=observe, test=test, plot=plot)
    cmds[MODE]()
