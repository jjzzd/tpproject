# A100 双卡多模态训练工程（可修改起点）

这是一个可运行、可阅读、方便替换模型的数据到训练闭环。项目针对两张 NVIDIA A100 80GB，包含：

- 图片 + 文本、纯文本、单图和多图 JSONL 输入；
- 无真实数据时的确定性合成数据；
- Tensor Parallel（TP2）；
- 两阶段 Pipeline Parallel（PP2）；
- FSDP2（对应参数、梯度和优化器状态分片）；
- PyTorch SDPA；
- Activation Checkpointing（PPT 中的 GC/重计算）；
- Fused AdamW；
- 可选 TorchAO 8-bit AdamW；
- BF16、梯度累积、梯度裁剪、峰值显存和吞吐日志；
- 单卡、TP2、PP2、FSDP2 的独立配置；
- 单卡/DDP、TP/FSDP 分布式 checkpoint 和 PP 分 stage checkpoint 保存。

当前内置模型是一个小型“视觉 patch token + 文本 token + Decoder Transformer”，主要用于跑通数据、并行和显存技术栈。它不是已经训练好的视觉语言模型。

## 1. 为什么并行方案是四套配置

两张 GPU 不能同时使用 TP2、PP2 和 FSDP2。GPU 数量大致满足：

```text
world_size = TP × PP × DP/FSDP
```

所以两张卡可以分别运行：

| 配置 | TP | PP | FSDP | GPU |
|---|---:|---:|---:|---:|
| `single.yaml` | 1 | 1 | 1 | 1 |
| `tp2.yaml` | 2 | 1 | 1 | 2 |
| `pp2.yaml` | 1 | 2 | 1 | 2 |
| `fsdp2.yaml` | 1 | 1 | 2 | 2 |

TP2+PP2 或 TP2+FSDP2 至少需要四张 GPU。工程会在启动时检查并拒绝错误组合。

## 2. 环境

建议环境：

- Linux；
- Python 3.11；
- PyTorch 2.5 或更新版本；
- CUDA 12.x；
- 两张卡之间最好有 NVLink；
- TP/FSDP 依赖 PyTorch DTensor、Tensor Parallel 和 FSDP2 API。

安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

使用 8-bit AdamW 时额外安装：

```bash
pip install -e ".[lowbit]"
```

先检查 GPU 拓扑：

```bash
nvidia-smi topo -m
```

TP 对 GPU 间带宽比较敏感。若两卡只有 PCIe 而没有 NVLink，TP 吞吐可能不理想，但显存分片仍然有效。

## 3. 先使用合成数据运行

单卡烟雾测试：

```bash
python -m mmtrain.train --config configs/single.yaml
```

双卡 TP：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m mmtrain.train --config configs/tp2.yaml
```

双卡 PP：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m mmtrain.train --config configs/pp2.yaml
```

双卡 FSDP2：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m mmtrain.train --config configs/fsdp2.yaml
```

每个训练 step 会向终端和 `training.output_dir/metrics.jsonl` 输出：

```json
{
  "step": 1,
  "loss": 5.61,
  "seconds": 0.82,
  "tokens_per_second": 1248.0,
  "peak_allocated_gib": 11.2,
  "strategy": "tp"
}
```

## 4. 接入你的数据

将数据和图片复制到：

```text
data/raw/
├── train.jsonl
└── images/
    ├── 000001.jpg
    └── 000002.png
```

### 4.1 图片描述/继续预训练格式

```json
{"text":"这张图片中有一台停在路边的汽车。","image":"images/000001.jpg"}
```

多图：

```json
{"text":"比较这两张图片的区别。","images":["images/000001.jpg","images/000002.png"]}
```

纯文本：

```json
{"text":"这是一条没有图片的训练文本。"}
```

### 4.2 指令微调格式

```json
{"prompt":"图片中有什么？","response":"图片中有一只猫。","image":"images/cat.jpg"}
```

也支持 messages：

```json
{
  "messages": [
    {"role":"system","content":"你是一个视觉助手。"},
    {"role":"user","content":"描述图片内容。"},
    {"role":"assistant","content":"图片中是一条河流。"}
  ],
  "image":"images/river.jpg"
}
```

训练前检查数据：

```bash
python -m mmtrain.inspect_data \
  --input data/raw/train.jsonl \
  --image-root data/raw
```

检查器会报告样本数、字符数、图片数、空文本和缺失图片。发现缺图或空文本时返回非零状态，适合放进提交训练任务前的脚本。

复制并修改配置：

```bash
cp configs/jsonl_example.yaml configs/my_train.yaml
```

至少修改：

```yaml
data:
  mode: jsonl
  path: data/raw/train.jsonl
  image_root: data/raw
```

然后使用对应启动命令，将 `--config` 指向新配置。

## 5. PPT 中技术的对应关系

| 技术 | 本工程位置 | 说明 |
|---|---|---|
| 8-bit Optimizer | `optimizer.type: adamw_8bit` | TorchAO AdamW8bit；与 Fused AdamW 二选一 |
| GC/重计算 | `training.activation_checkpointing` | 使用 non-reentrant activation checkpoint |
| SDPA | `SelfAttention` | 调用 `scaled_dot_product_attention` |
| Fused AdamW | `optimizer.type: fused_adamw` | CUDA 上使用 `torch.optim.AdamW(fused=True)` |
| TP | `configs/tp2.yaml` | Q/K/V、输出投影和 MLP 矩阵分片 |
| PP | `configs/pp2.yaml` | 两个 stage，手工 send/recv，GPipe fill-drain |
| ZeRO/FSDP | `configs/fsdp2.yaml` | PyTorch FSDP2 按层分片 |

切换到 8-bit AdamW：

```yaml
optimizer:
  type: adamw_8bit
  lr: 0.0003
  weight_decay: 0.1
  betas: [0.9, 0.95]
```

不要同时配置 Fused AdamW 和 8-bit AdamW；它们是两个优化器实现。

## 6. 模型数据流

```text
N 张图片
  ↓ resize + normalize
Conv2d patch embedding
  ↓
图片 token ─────┐
                ├─ 拼接 + 位置编码 → Decoder Transformer → LM logits
文本 token ─────┘                                      ↓
                                             只对文本区域计算 loss
```

图片区域使用双向可见 mask；文本区域可以看到全部有效图片 token 和之前的文本 token。填充图片和填充文本会被 mask。

## 7. 如何替换为真实预训练 VLM

当前最需要替换的是：

1. `ByteTokenizer`：替换为目标模型原生 tokenizer。
2. `MultimodalEmbedding`：替换为 CLIP/SigLIP 或目标模型的视觉编码器和 projector。
3. `MultimodalTransformer`：替换为 Qwen2-VL、LLaVA、InternVL 等模型实现。
4. 数据 collator：按目标模型 chat template 生成 `input_ids` 和监督 mask。
5. checkpoint 导入/导出：增加 Hugging Face 权重与 DCP 权重转换。

建议保留下面的工程接口不变：

- `build_dataset()`；
- 配置文件里的 `parallel`、`optimizer`、`training`；
- 训练指标 JSONL；
- 单卡/TP/PP/FSDP 的启动入口；
- 数据预检查脚本。

这样替换模型后不需要重写任务提交和实验管理代码。

## 8. 当前明确不足（接入正式训练前请处理）

### 模型和 tokenizer

- 内置模型是教学/验证模型，不是预训练 VLM，不能直接获得高质量图文能力。
- 内置字节 tokenizer 只有 260 个 token，中文也能编码，但序列效率较低。
- 没有加载 Qwen2-VL/LLaVA 等预训练 checkpoint。
- 当前视觉编码器只是 patch Conv2d，没有使用预训练 CLIP/SigLIP。
- 视觉位置编码比较简单，未实现原图比例、动态分辨率或 RoPE。

### 数据

- JSONL 当前一次性加载到内存；超大数据集应改为索引文件、WebDataset、Parquet 或流式读取。
- 图片统一 resize 成正方形，可能产生形变；正式工程应增加保持比例的 resize/crop 策略。
- 每条样本最多读取 `model.max_images` 张图片，超出部分会截断。
- messages 中嵌套的 image 项不会自动读取路径；图片路径应同时放在顶层 `image`/`images` 字段。
- 目前把整段 messages 拼成文本并对全部文本计算 loss；正式 SFT 应只监督 assistant 回答区域。
- 没有数据去重、图像损坏隔离、数据质量评分、训练/验证集自动拆分。

### TP

- TP 分片了 Transformer 主体的 Attention/MLP 权重，但 embedding、vision encoder、norm 和 lm_head 当前复制在两卡上，所以显存不会严格减半。
- 尚未实现 Sequence Parallel；长序列时仍有进一步降低激活显存的空间。
- 当前 TP 是为两卡设计的最小实现，升级 PyTorch 后应重新跑数值一致性测试。

### PP

- PP 当前固定两个 stage。
- 当前调度是 GPipe fill-drain，不是高效率 1F1B，因此会有流水线气泡，并保存全部 microbatch 的中间激活。
- 分层只按层数切分，没有根据 vision encoder、embedding、lm_head 的真实显存做自动均衡。
- 两个 rank 各自读取同一批数据，便于实现和校验，但会重复数据读取。
- 后续应迁移到 PyTorch PipelineSchedule1F1B 或 TorchTitan 的 pipeline 实现。

### FSDP/ZeRO

- 工程实现的是 FSDP2，没有单独依赖 DeepSpeed ZeRO。
- 未实现 CPU offload；A100 80GB 双卡一般应先避免 offload，以免 PCIe 成为瓶颈。
- TP/FSDP 使用 Distributed Checkpoint 保存，但本版本尚未实现统一的 `--resume` 加载入口。

### 优化器和 checkpoint

- TorchAO 8-bit AdamW 与 DTensor/FSDP 的组合随 PyTorch/TorchAO 版本变化，正式使用前必须逐个配置跑测试。
- PP checkpoint 是每个 stage 一个文件；改变 PP 切分后不能直接恢复。
- 没有 checkpoint 保留策略、异步保存、对象存储上传和训练故障自动重启。

### 训练功能

- 没有学习率 warmup/cosine scheduler、验证循环、生成评测、W&B/TensorBoard 集成。
- 没有 LoRA/QLoRA；当前是全参数训练。
- 没有 FP8、FlashAttention 版本锁定、通信计算重叠或自定义 fused kernel。
- 多节点配置尚未提供，只包含单机两卡启动脚本。
- 尚未在你的服务器驱动、CUDA 和具体 PyTorch 版本上执行 A100 实机验证。

这些限制被明确保留，是为了让第一版代码足够小、可审计。建议先在服务器上用合成数据跑通四套配置，再选择目标预训练模型进行第二阶段改造。

## 9. 验证建议

先运行单元测试：

```bash
pip install -e ".[dev]"
pytest -q
```

再依次执行：

1. `single.yaml` 运行 20 step；
2. `tp2.yaml` 运行 20 step；
3. `pp2.yaml` 运行 20 step；
4. `fsdp2.yaml` 运行 20 step；
5. 比较每套配置的 loss、峰值显存和 tokens/s；
6. 将 `data.mode` 切换为 `jsonl`，用 10～100 条真实数据做 smoke test；
7. 最后逐步放大层数、hidden size、图片分辨率和文本长度。

生产实验中不要一次改多个优化项。建议分别记录：

- Fused AdamW vs 8-bit AdamW；
- Activation Checkpointing 开/关；
- TP2 vs PP2 vs FSDP2；
- 图片分辨率 224/336/448；
- 文本长度 512/2048/4096。

## 10. 主要入口

- `src/mmtrain/train.py`：训练入口和 PP 调度；
- `src/mmtrain/model.py`：多模态模型、SDPA 和 checkpointing；
- `src/mmtrain/parallel.py`：TP、FSDP2 和 DDP；
- `src/mmtrain/data.py`：JSONL/图片/合成数据；
- `src/mmtrain/inspect_data.py`：数据检查；
- `configs/`：各并行策略配置。

## 11. GitHub 与服务器自动同步

仓库地址：

```text
https://github.com/jjzzd/tpproject.git
```

服务器首次部署：

```bash
cd ~/Sata2/sjw
git clone https://github.com/jjzzd/tpproject.git multimodal-training
cd multimodal-training
bash scripts/install_server_sync.sh
```

安装后，服务器每分钟运行一次 `scripts/sync_from_git.sh`。该脚本执行：

1. 检查 Git 管理的代码文件是否有服务器本地改动；
2. 执行 `git fetch origin main`；
3. 只接受 fast-forward 更新；
4. 将输出记录到 `logs/git-sync.log`。

它不会执行 `git reset --hard`，所以不会静默覆盖服务器上的代码修改。如果服务器代码有未提交修改，会停止同步并记录错误。数据、日志和 checkpoint 位于 `.gitignore` 覆盖范围内，不会随代码拉取被删除。

检查同步：

```bash
crontab -l

```

卸载自动同步：

```bash

```

如果仓库是私有仓库，服务器还需要配置只读 GitHub deploy key；不要把私钥、PAT 或密码写入本仓库。
