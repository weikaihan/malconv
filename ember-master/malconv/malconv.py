# 由于在Windows系统下，dataloader在进行num_workers参数配置时需要在主模块中使用 if __name__ == '__main__': 来包装代码，
# 每个子进程都会重新导入主模块，如果没有if __name__ == '__main__':保护，会导致递归创建进程
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         模 型 定 义                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class MalConv1(nn.Module):
    """
    CNN下采样 + Transformer编码器

    为什么这样设计？

    问题: 1MB的PE文件 = 100万字节。Transformer的自注意力是O(n²)，
          100万² = 1万亿次计算，不可行。

    解法: 用跨步卷积先把序列"压缩"到2000个位置(和原始MalConv一样)，
          然后在这个可管理的长度上用Transformer捕捉全局依赖。

    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐
    │ 原始字节 │ → │ Embedding │ → │ 跨步卷积  │ → │ 残差精炼 │ → │Trans-│
    │ 1MB      │    │ 257→128  │    │ 100万→2K │    │ 局部平滑 │    │former│
    └─────────┘    └──────────┘    └──────────┘    └──────────┘    └──────┘
                                                                       │
    ┌──────┐    ┌──────────┐    ┌──────────┐                           │
    │ 恶意 │ ← │ 分类头   │ ← │ 注意力    │ ←─────────────────────────┘
    │/良性 │    │          │    │ 池化     │     (B, 2000, 128)
    └──────┘    └──────────┘    └──────────┘

    每个组件的设计意图:

    1. Embedding:   字节值0-255是离散符号，没有大小关系。Embedding把它们
                    映射为128维稠密向量，让模型学习"MZ"(0x4d 0x5a)这种
                    相邻字节之间的关系。

    2. 跨步卷积:    核大小=步长=500，意味着每500个字节打包成一组，完全不重叠。
                    这和原始MalConv一样。1MB / 500 ≈ 2000个"块"。
                    你可以想象成把PE文件切成2000段，每段500字节。

    3. 残差精炼:    两个3x1卷积+残差连接。跨步卷积不重叠，相邻"块"的边界
                    可能正好切断一个关键结构(比如嵌在边界上的4字节地址)。
                    这个精炼步骤让每个位置融合左右邻居的信息，修复边界效应。

    4. Transformer: 4层多头自注意力。这是整个模型的"思考"核心。
                    PE header在位置0, import table在位置1500, 它们是相关的
                    (header声明了导入表的RVA)。卷积只能看到局部，
                    自注意力让位置0和位置1500直接对话。

    5. 注意力池化:  原始MalConv用GlobalMaxPooling——取每个特征维度的最大值。
                    这很暴力。注意力池化让模型自己学"这段PE里哪几个块最关键"，
                    比如entry point附近的块权重可能更高。

    6. 分类头:      两层MLP，输出一个logit。>0是恶意，<0是良性。
                    配合BCEWithLogitsLoss训练。

    参数说明:
        maxlen:   最大输入长度(字节数), 默认2^20≈1MB。
                  超过的被截断，不足的末尾填256(padding)。
        input_dim: 词汇表大小=257 (0-255字节值 + 256作padding标记)
        embed_dim: 每个字节被映射成多少维向量
        conv_filters: 跨步卷积的输出通道数
        conv_kernel:  跨步卷积的核大小(也是"块"的大小)
        conv_stride:  跨步卷积的步长(也是"块"之间的间隔)
        d_model:  Transformer内部维度
        nhead:    多头注意力的头数(d_model必须能被nhead整除)
        num_layers: Transformer层数
        d_ff:    Transformer前馈网络的隐藏层维度
        dropout:  Dropout概率，防止过拟合
        num_classes: 输出类别数, 1=二分类(恶意/良性), 9=多标签(9个家族)
    """

    def __init__(
        self,
        maxlen: int = 2 ** 20,
        input_dim: int = 257,
        embed_dim: int = 128,
        conv_filters: int = 256,
        conv_kernel: int = 500,
        conv_stride: int = 500,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()

        # === 第1步: 字节Embedding ===
        # 257个token(0-255字节 + padding), 每个映射为128维向量。
        # padding_idx=256: padding位置的embedding固定为0向量，不参与梯度。
        # 类比NLP: 就像Word2Vec, 但"词表"是所有可能的字节值。
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=256)

        # === 第2步: CNN下采样 ===
        # 这是解决"100万token太长"的关键。
        # Conv1d的输入是(B, C, L): Batch × Channel × Length
        # 经过此层: (B, 256, ~2000)   意味着100万个字节被压缩到2000个"超像素"
        #
        # 注意: 这里没有用Gated Attention(原始MalConv的双路卷积+sigmoid门)，
        # 而是用BatchNorm+ReLU。原因是Transformer有更强的表达能力，
        # 不再需要CNN侧的门控机制来选择性放大特征。
        self.conv_down = nn.Conv1d(
            embed_dim, conv_filters,
            kernel_size=conv_kernel, stride=conv_stride,
        )
        self.bn_down = nn.BatchNorm1d(conv_filters)

        # === 第3步: 残差CNN精炼块 ===
        # 跨步卷积分割的块之间没有重叠。如果PE的某个关键4字节字段
        # 正好跨越了两个块的边界，信息就被切断了。
        # 这两个残差卷积让相邻块的表示互相"渗透"，修复边界信息损失。
        self.conv_res1 = nn.Conv1d(conv_filters, conv_filters,
                                   kernel_size=3, padding=1)
        self.bn_res1 = nn.BatchNorm1d(conv_filters)
        self.conv_res2 = nn.Conv1d(conv_filters, conv_filters,
                                   kernel_size=3, padding=1)
        self.bn_res2 = nn.BatchNorm1d(conv_filters)

        # === 第4步: 投影到Transformer维度 ===
        # 如果conv_filters != d_model, 需要一个线性映射。
        # 128是Transformer的标准小维度, 256是CNN提取丰富特征需要的维度。
        # 两者不一定相等，这里做转换。
        if conv_filters != d_model:
            self.proj = nn.Linear(conv_filters, d_model)
        else:
            self.proj = nn.Identity()

        # === 第5步: 位置编码 ===
        # Transformer本身不知道"顺序"。需要告诉它第0个位置是PE头部,
        # 第2000个位置是文件末尾。
        #
        # 方案: 可学习的位置编码(每次训练自己学)
        # 备选: 正弦位置编码(Transformer原论文, 不需要学习)
        # 这里用可学习的，因为PE的结构位置是固定的——PE header永远在开头。
        #
        # seq_len的计算:
        #   L_out = ⌊(maxlen - kernel_size) / stride⌋ + 1
        #   例: maxlen=1,048,576, kernel=500, stride=500
        #       → (1,048,576 - 500) // 500 + 1 = 2097
        seq_len = (maxlen - conv_kernel) // conv_stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # === 第6步: Transformer编码器 ===
        # 标准实现。batch_first=True表示输入是(B, L, D)而非(L, B, D)。
        # GELU而非ReLU: 现代Transformer的标准选择，在0附近更平滑。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === 第7步: 注意力池化 ===
        # 问题: Transformer输出是(B, 2000, 128), 怎么变成(B, 128)?
        #
        # 方案A: Mean Pooling — 所有位置一视同仁
        # 方案B: Max Pooling  — 原始MalConv用的方式
        # 方案C: 注意力池化 — 让模型自己学哪个位置重要 ← 我们选这个
        #
        # 具体做法: 学一个可训练的"query"向量, 对每个位置的128维表示打分,
        # 然后softmax归一化, 加权求和。
        self.attn_query = nn.Linear(d_model, 1)

        # === 第8步: 分类头 ===
        # 二分类模式(num_classes=1): 输出单个logit, >0恶意
        # 多标签模式(num_classes=9): 输出9个logit, 分别对应9个恶意家族
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """合理的初始化保证训练初期梯度不爆炸也不消失"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m is not self.attn_query:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif m is self.attn_query:
                # 注意力查询向量初始化为0附近的小值,
                # 保证训练初期各位置注意力均匀分布
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, maxlen) — 整数张量, 每个位置是一个字节值(0-256)。
               B = batch_size

        Returns:
            logits: (B, num_classes) — 未经过sigmoid的分数。
                    二分类时(B,1), >0表示恶意。
                    多标签时(B,9), 每列对应一个家族, >0表示属于该家族。
        """
        # [1] Embedding
        # (B, maxlen) → (B, maxlen, embed_dim)
        x = self.embed(x)

        # [2] CNN需要(B, C, L)格式
        x = x.permute(0, 2, 1)          # → (B, embed_dim, maxlen)

        # [3] 跨步卷积下采样
        # (B, embed_dim, 1,048,576) → (B, conv_filters, ~2097)
        x = self.conv_down(x)
        x = F.relu(self.bn_down(x))

        # [4] 残差精炼 (修复块边界信息损失)
        identity = x
        x = F.relu(self.bn_res1(self.conv_res1(x)))
        x = self.bn_res2(self.conv_res2(x))
        x = F.relu(x + identity)         # 残差: 精炼后的特征 + 原始特征

        # [5] 转回(B, L, C)给Transformer
        x = x.permute(0, 2, 1)          # → (B, seq_len, conv_filters)

        # [6] 投影到d_model
        x = self.proj(x)                 # → (B, seq_len, d_model)

        # [7] 加位置编码
        # 只取前seq_len个位置(防止maxlen变化时维度不匹配)
        x = x + self.pos_embed[:, :x.size(1), :]

        # [8] Transformer编码
        x = self.transformer(x)          # → (B, seq_len, d_model)

        # [9] 注意力池化
        # 对每个位置打分 → softmax → 加权求和
        attn_scores = self.attn_query(x)              # (B, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, seq_len, 1)
        x = (x * attn_weights).sum(dim=1)             # (B, d_model)

        # [10] 分类
        logits = self.classifier(x)      # → (B, num_classes)
        return logits


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                       数 据 加 载                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class ByteDataset(Dataset):
    """
    字节序列数据集

    从文件路径列表中读取样本。
    无论原始文件是 .exe 还是经过 latin-1 转换的 .txt, 都用 "rb" 模式读取,
    读出来的字节序列完全一致, 模型无感知。

    Args:
        file_paths: 样本文件的绝对路径列表
        labels:     标签列表, 0.0=良性, 1.0=恶意
        maxlen:     截断/补齐的目标长度
        padding_char: 填充值(默认256, 对应Embedding的padding_idx)
    """

    def __init__(self, file_paths, labels, maxlen=2 ** 20, padding_char=256):
        self.file_paths = list(file_paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.maxlen = maxlen
        self.padding_char = padding_char

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # 二进制读取: 对 .exe 和 latin-1 .txt 效果完全一样
        with open(path, "rb") as f:
            raw = f.read()

        # 截断/补齐
        buf = np.ones((self.maxlen,), dtype=np.int64) * self.padding_char
        chunk = np.frombuffer(raw[:self.maxlen], dtype=np.uint8)
        buf[: len(chunk)] = chunk

        return (
            torch.from_numpy(buf),       # (maxlen,)  int64
            torch.tensor(label, dtype=torch.float32),  # 标量
        )


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     工 具 函 数 (数据收集)                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# 9个恶意软件家族 — 多标签分类的输出维度
FAMILIES = [
    "generic", "trojan", "ransomware", "worm", "backdoor",
    "spyware", "rootkit", "encrypter", "downloader",
]


def gather_files_from_dir(directory, extensions=None):
    """递归收集目录下所有文件路径"""
    paths = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if extensions is None or any(fname.lower().endswith(ext) for ext in extensions):
                paths.append(os.path.join(root, fname))
    return paths


def load_multi_label_data(benign_dir, malware_dir, labels_dir,
                          families=FAMILIES, malware_threshold=0.5):
    """
    从CSV标签文件中加载多标签数据

    流程:
      1. 读取 labels/benign.csv 和 labels/malware.csv
      2. 构建 hash → 多标签向量 的映射
      3. 扫描良性/恶意目录, 匹配文件到标签

    Args:
        benign_dir:  良性样本目录
        malware_dir: 恶意样本目录
        labels_dir:  CSV标签目录
        families:    家族列表, 默认9个
        malware_threshold: malice分数阈值, >=此值才算恶意

    Returns:
        file_paths: [(path, is_malware_bool), ...]
        multi_labels: (N, len(families)) np.float32 数组
        malice_scores: (N,) np.float32 数组 (用于分层采样)
    """
    labels_path = os.path.join(labels_dir, "benign.csv")
    malware_csv = os.path.join(labels_dir, "malware.csv")

    if not os.path.exists(labels_path) or not os.path.exists(malware_csv):
        raise FileNotFoundError(
            f"找不到标签CSV文件: {labels_path} 或 {malware_csv}"
        )

    df_benign = pd.read_csv(labels_path)
    df_malware = pd.read_csv(malware_csv)

    # 构建 hash → 家族标签映射
    hash_to_families = {}
    hash_to_malice = {}

    for _, row in df_benign.iterrows():
        h = str(row["hash"]).lower()
        hash_to_families[h] = [0.0] * len(families)
        hash_to_malice[h] = 0.0

    for _, row in df_malware.iterrows():
        h = str(row["hash"]).lower()
        malice = float(row.get("malice", 1.0))
        if malice < malware_threshold:
            continue  # 跳过恶意程度不够的样本
        hash_to_families[h] = [float(row.get(fam, 0)) for fam in families]
        hash_to_malice[h] = malice

    # 扫描目录, 匹配文件
    file_paths = []
    multi_labels = []
    malice_list = []

    for dir_path, is_malware in [(benign_dir, False), (malware_dir, True)]:
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            # 文件名格式: {hash}.txt 或 {hash}
            h = os.path.splitext(fname)[0].lower()
            if h not in hash_to_families:
                continue
            file_paths.append(fpath)
            multi_labels.append(hash_to_families[h])
            malice_list.append(hash_to_malice.get(h, float(is_malware)))

    if not file_paths:
        raise RuntimeError("未找到任何匹配的样本文件, 请检查目录和CSV文件")

    return (
        file_paths,
        np.array(multi_labels, dtype=np.float32),
        np.array(malice_list, dtype=np.float32),
    )


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         主 入 口                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    # ==============================
    # 1. 环境准备 & 超参数配置
    # ==============================
    # 数据集路径
    benign_dir = "./datasets/data/benign"
    malware_dir = "./datasets/data/malware"
    labels_dir = "./datasets/labels"

    # 二分类: 良性 vs 恶意
    num_classes = 1                    # 1 = 二分类(恶意/良性)

    # 数据参数
    maxlen = 2 ** 20
    test_size = 0.1
    val_size = 0.1
    batch_size = 4             # 翻倍加速, (4*1M*128)fp16 ≈ 1GB 显存
    grad_accum_steps = 8       # 等效 batch = 4*8 = 32
    num_workers = 0 if os.name == "nt" else 4

    # 训练参数
    epochs = 15
    lr = 5e-4                 # 稍降学习率, 配合大pos_weight更稳定
    weight_decay = 1e-4

    # 模型参数
    embed_dim = 64            # 原论文embedding=8, 64已足够表示256种字节
    conv_filters = 128        # 原论文filters=128, 1万样本不需要256通道
    conv_kernel = 500
    conv_stride = 500
    d_model = 96              # 96/4=24 每头注意力维度, 足够捕捉PE结构依赖
    nhead = 4
    num_layers = 2
    d_ff = 256
    dropout = 0.2

    # 随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ==============================
    # 2. 从CSV加载多标签数据
    # ==============================
    print(f"\n加载数据集...")
    file_paths, multi_labels, malice_scores = load_multi_label_data(
        benign_dir, malware_dir, labels_dir, families=FAMILIES,
        malware_threshold=0.5,
    )

    # 多标签 → 二分类: 任意家族>0 即为恶意
    binary_labels = (multi_labels.sum(axis=1) > 0).astype(np.float32)

    num_benign = int((binary_labels == 0).sum())
    num_malware = int((binary_labels == 1).sum())
    print(f"良性样本: {num_benign}")
    print(f"恶意样本: {num_malware}")

    # malice 二值化用于分层采样
    stratify_labels = (malice_scores >= 0.5).astype(int)

    # ==============================
    # 3. 划分数据集
    # ==============================
    train_paths, test_paths, train_labels, test_labels, _, test_strat = train_test_split(
        file_paths, binary_labels, stratify_labels,
        test_size=test_size, random_state=seed, stratify=stratify_labels,
    )

    val_ratio = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_ratio, random_state=seed, stratify=train_labels,
    )

    print(f"训练/验证/测试: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")

    # ==============================
    # 4. 构建 DataLoader
    # ==============================
    train_set = ByteDataset(train_paths, train_labels, maxlen=maxlen)
    val_set   = ByteDataset(val_paths,   val_labels,   maxlen=maxlen)
    test_set  = ByteDataset(test_paths,  test_labels,  maxlen=maxlen)

    # 加权采样器: 良性样本过采样, 解决10:1不平衡
    is_benign_train = (train_labels == 0)
    n_benign = is_benign_train.sum()
    n_mal = len(train_labels) - n_benign
    weights = np.ones(len(train_labels), dtype=np.float64)
    if n_benign > 0:
        weights[is_benign_train] = n_mal / max(n_benign, 1)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_labels), replacement=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ==============================
    # 5. 构建模型 (多标签模式: num_classes=9)
    # ==============================
    model = MalConv1(
        maxlen=maxlen,
        embed_dim=embed_dim,
        conv_filters=conv_filters,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    seq_len = (maxlen - conv_kernel) // conv_stride + 1
    print(f"参数量: {total_p:,}")
    print(f"输出维度: {num_classes} (二分类: 0=良性, 1=恶意)")
    print(f"压缩后序列长度: {seq_len}")

    # ==============================
    # 6. 优化器 & 类别加权损失
    # ==============================
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )   # val loss不降时才降LR, 不浪费时间

    # 二分类 pos_weight: 恶意数/良性数 ≈ 9.5, 让两个类平衡
    n_pos = int(train_labels.sum())         # 恶意样本数
    n_neg = len(train_labels) - n_pos       # 良性样本数
    pw = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"良性样本权重: {pw:.1f} (良性样本少, loss权重×{pw:.1f})")

    # ==============================
    # 7. 训练/评估函数
    # ==============================
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        optimizer.zero_grad()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(x).squeeze(-1)      # (B,)
                loss = criterion(logits, y) / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            n += x.size(0)

        return total_loss / n, correct / n

    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        n = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            n += x.size(0)

        return total_loss / n, correct / n

    # ==============================
    # 8. 训练循环 (记录历史用于可视化)
    # ==============================
    best_val_acc = 0.0
    best_epoch = 0
    output_model_path = "./malconv1_best.pt"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n开始训练...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"LR {lr_now:.2e} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_model_path)
            print(f"  [*] 最佳模型已保存 (Val Acc {val_acc:.4f})")

    # ==============================
    # 9. 最终测试 — 每家族详细指标
    # ==============================
    model.load_state_dict(torch.load(output_model_path, map_location=device))
    print(f"\n{'='*50}")
    print(f"加载最佳模型 (Epoch {best_epoch}, Val Acc {best_val_acc:.4f})")
    print(f"{'='*50}")

    # 收集测试集全部预测
    model.eval()
    all_preds = []
    all_labels_test = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(x).squeeze(-1)
            all_preds.append((torch.sigmoid(logits) >= 0.5).float().cpu().numpy())
            all_labels_test.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels_test = np.concatenate(all_labels_test)

    tp = int(((all_preds == 1) & (all_labels_test == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels_test == 0)).sum())
    tn = int(((all_preds == 0) & (all_labels_test == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels_test == 1)).sum())

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"混淆矩阵:")
    print(f"              预测良性  预测恶意")
    print(f"  实际良性:     {tn:5d}      {fp:5d}")
    print(f"  实际恶意:     {fn:5d}      {tp:5d}")
    print(f"\n  准确率 Accuracy:  {acc:.4f}")
    print(f"  精确率 Precision: {prec:.4f} (预测为恶意中真正恶意的比例)")
    print(f"  召回率 Recall:    {rec:.4f} (实际恶意中被找出的比例)")
    print(f"  F1 分数:          {f1:.4f}")

    # ---- 保存训练曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(history["train_loss"], "b-o", markersize=4, label="Train Loss")
        axes[0].plot(history["val_loss"], "r-o", markersize=4, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["train_acc"], "b-o", markersize=4, label="Train Acc")
        axes[1].plot(history["val_acc"], "r-o", markersize=4, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (per-label)")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("MalConv1 — Multi-Label Malware Detection", fontsize=13)
        plt.tight_layout()
        plt.savefig("./training_curve.png", dpi=150, bbox_inches="tight")
        print("\n训练曲线已保存至: ./training_curve.png")
        plt.close()
    except Exception as e:
        print(f"\n(可视化保存失败: {e})"
              f"\n  安装: pip install matplotlib")

    pd.DataFrame(history).to_csv("./training_history.csv", index=False)
    print("训练历史已保存至: ./training_history.csv")

    return model


if __name__ == "__main__":
    main()