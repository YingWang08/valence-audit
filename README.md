# 人–AI 价值不对称审计 · 一键实验工程

测量大语言模型在「人类 vs. AI」评价上的系统性**价值不对称**（machine-deference），
并按维度（H2）与对齐阶段（H3）拆解，附对冲/拒答分析（H5）。理论框架：安德斯「普罗米修斯式羞愧」
（写进论文 intro/discussion，**不在代码里**）。本工程用 **NVIDIA NIM 免费 API**，**无需 GPU、零额外花费**。

> 本工程的整条流水线已用 `--mock` 离线跑通（评分 cell 配对 + 文本测量 + 收敛效度 + 4 张图全部产出）。
> 你要做的就是：装依赖 → 填密钥 → 按顺序跑三个模式。

---

## ⚠️ 第 0 步：安全（务必先做）

如果你在任何地方贴过你的 `nvapi-` 密钥，**先去 build.nvidia.com → API Keys 把旧的吊销、生成一把新的**。
新密钥只放进本地 `.env`，**不要贴进聊天、截图、git、文档**。

---

## 1. 安装

需要 Python 3.10+。在 PyCharm 里打开本文件夹，然后终端执行：

```bash
# 建虚拟环境（可选但推荐）
python -m venv .venv
# 激活： Windows: .venv\Scripts\activate   Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

只装核心依赖即可全程运行（全部能离线工作）。`transformers/torch`（M2）和 `pymer4/bambi`（交叉随机效应）
是**可选**的，到论文阶段再装（见 §8、§9）。

## 2. 配置密钥

```bash
# 复制模板为 .env，填入你的新密钥
cp .env .env        # Windows: copy .env .env
# 然后编辑 .env：NVIDIA_API_KEY=nvapi-你的新密钥
```

PyCharm 里也可用 Run → Edit Configurations → Environment variables 设 `NVIDIA_API_KEY`。

---

## 3. 三种运行模式（按顺序来）

```bash
# ① 离线干跑：不需密钥/网络，几十秒，验证整条流水线在你机器上能跑通
python run_all.py --mock

# ② 看你的密钥能调哪些模型（关键！决定 H3 能不能做）
python run_all.py --list-models

# ③ 冒烟测试：真实 API，1 模型 1 维度 2 重复，最省额度，验证 key+连通
python run_all.py --smoke

# ④ 跑完整实验
python run_all.py --full
```

**强烈建议**：①→②→③ 全过了，再跑 ④。`--mock` 证明代码没问题；`--smoke` 证明你的 key 通。
直接上 ④ 万一中途出错会浪费时间和额度。`--full` 支持**断点续跑**，中断后重跑会跳过已完成的样本。

---

## 4. 输出说明（`data/results/`）

| 文件 | 内容 | 对应 |
|---|---|---|
| `summary.csv` | H1 总体不对称 + CI、H3 关联值 | H1/H3 |
| `H2_by_dimension.csv` | 各维度不对称 + 95%CI + FDR 校正 + **标准化效应量 d** + 期望方向 | **H2** |
| `H5_hedge_rate.csv` / `H5_refuse_rate.csv` | 按对齐阶段的对冲/拒答率 | **H5** |
| `convergence.csv` | rating↔M1（及 M1↔M2）在 model×dim×lang cell 上的相关 = 收敛效度 | 测量信度 |
| `by_family.csv` | 各架构家族(model_family)的不对称均值（看模型非独立性） | 稳健性 |
| `robustness_loo.csv` | 留一模型后的 H1 | 稳健性 |
| `fig2_per_model.png` | 分模型森林图 | H1 |
| `fig3_by_dimension.png` | 维度梯度（蓝=抬机器/红=抬人） | H2 |
| `fig4_base_vs_instruct.png` | base vs instruct 对比（若有 base） | H3 |
| `fig5_hedge_rate.png` | 对冲率 | H5 |

原始数据：`data/raw/<model>.jsonl`（每条调用的完整参数+回复，可复现/可复查）。
中间测量：`data/measured/items.parquet`（每行一个项目级观测）。

---

## 5. 关于 base 模型与 H3（重要 · 诚实说明）

H3（base vs instruct 对照）是创新点之一，但**它需要 base（未对齐）模型，而 NVIDIA NIM 主要托管 instruct 模型**。
跑完 `--list-models` 后：

- **列表里有 base 模型** → 填进 `config/models.yaml` 的 base 段（已留注释位），H3 完整可做。
- **没有 base 模型**（很可能）→ 代码会自动跳过 H3 并提示，**H1/H2/H5 完全不受影响**。
  此时论文里把 H3 处理成二选一：
  1. **退为探索性分析**：比较可用 instruct 模型中「对齐强度/家族」的差异（关联、非因果），明确标注为探索性；或
  2. **用 H5 承载对齐效应**：对冲/拒答率随对齐阶段的变化本身就是「对齐影响人-机评价表达」的证据（mock 里 instruct 对冲率 30% vs base 10%，这条很有说服力）。

无论哪种，**你的核心贡献（H1 人-机评价轴 + H2 维度结构 + 安德斯框架）都成立**，不依赖 base 模型。

---

## 6. 调参（卡 / 额度有限怎么办）

改 `config/experiment.yaml`，不用动代码：

- 网络「很卡」→ `api.rate_per_sec: 1`、`api.concurrency: 2`。慢就让它挂着跑，断了重跑接着续。
- 额度有限 → 调小 `generation.repeats`（如全改 3）、删 `config/models.yaml` 里几个模型、`--full` 前先 `--smoke`。
- 想加维度 → 在 `config/dimensions.yaml` 加一项（给 en/zh 的 attr 与 skill）。
- 想加改写模板（压提示伪影）→ 在 `src/build_prompts.py` 的 `FRAMES` 里加句子。

---

## 7. 主分析的统计口径

- 主指标 = **rating**（评分式，最干净——逐个询问、无归因、无比较句问题）；没有 rating 时自动退回 text_m1。
- **评分式不对称按 cell 聚合后差分**：在 model×stage×dim×lang×template 的 cell 内，对 human/AI 各取均值再相减，
  消除旧版“按重复下标人为配对”的问题。
- H1 与 H2 **统一口径**：以**“模型”为重抽样单元的 cluster bootstrap**（给均值/95%CI/双侧 p），
  分析单元落在模型而非题项，从根上避免**伪重复**；H2 再做 **FDR（BH）** 校正，并报**标准化效应量 d**。
  （这比对每个题项做 t 检验诚实得多，也比 statsmodels 混合模型在“模型组很少”时稳健——后者会把 CI 炸到百万级。）
- H3：base/instruct 子集，**关联措辞**（代码注释与输出都强调非因果）；缺 base 时干净跳过，不影响 H1/H2/H5。
- H5：对冲/拒答率按对齐阶段。
- 收敛效度：rating↔M1（开 M2 后再加 M1↔M2）在 **model×dim×lang cell** 上的 Pearson/Spearman
  （不再按不可比的 template 下标错位 join）。
- 归因用**词边界匹配**（修复旧版 `"ai"` 子串误命中 explain/remain/… 的 bug）；含双方的比较句默认丢弃并计数，
  可在 `experiment.yaml` 置 `measurement.comparative_attribution: true` 启用保守的方向归因做稳健性检验。
- **所有丢弃率/解析失败率都会在 measure 阶段打印**，请如实写进 Methods/Limitations。

## 8. M2 transformer（论文阶段再开，强化收敛效度）

`config/experiment.yaml` 设 `measurement.enable_m2: true`，并 `pip install transformers torch`。
首次运行会联网下载模型（英文 cardiffnlp、中文 uer，几百 MB，CPU 慢但短文本可接受）。
开了之后 `convergence.csv` 会多出 M1↔M2 一致性——这是回应审稿「construct validity」的关键弹药。
若下载失败/没装 torch，代码会**自动退回 M1 并提示**，不会崩。

## 9. 交叉随机效应（最终论文用这个）

Python 端是开发级（cluster bootstrap，稳健但功效有限、只把“模型”当随机效应）。**论文显著性请以 R 的 lme4
交叉随机效应为准**——把**模型**与**措辞框架 frame**（`frame = format|language|template`，即同一句式的 item
随机效应）同时作为随机效应，功效更高也最贴合本设计：

```
asymmetry ~ dimension + language [+ alignment_stage] + (1|model) + (1|frame)
```

（dimension 是固定效应，故**不要**再加 `(1|dimension)`，会共线。）已附 `analyze_lme4.R`：

```bash
# 需要本机装 R 与包： install.packages(c("lme4","lmerTest"))
Rscript analyze_lme4.R              # 默认主指标 rating；text_m1 复核： Rscript analyze_lme4.R text_m1
```

measure 阶段已同时导出 `data/measured/items.csv`，R 端直接读 csv，**无需安装 arrow**。
H2 在 R 内同样做 BH 校正并核对方向假设；缺 base 时 H3 自动跳过。或纯 Python：装 `bambi`（贝叶斯）或 `pymer4`。

---

## 10. 常见问题排查

- **AuthenticationError** → `.env` 里密钥错/旧密钥已吊销。换新密钥。
- **某模型一直「跳过：模型不可用」** → 该模型 ID 在 NIM 上不存在。用 `--list-models` 看正确 ID，改 `models.yaml`。
- **rating 未配对丢弃很多 / 文本测量很少** → 模型没按要求只回数字，或自由文本把人和 AI 写在同一句（被归因丢弃）。
  对策：评分式提示已加「只回答数字」；自由文本归因的「比较句」问题见 §11。解析失败率会在 `[measure]` 行打印，**写进论文 Methods 别瞒**。
- **中文分词第一次慢** → jieba 首次加载词典，正常。

## 11. 已知限制 / 可改进（也要写进论文 Methods 与 Limitations）

- **归因的比较句问题**：自由文本 M1 按句归因；含双方的比较句默认丢弃并计数。
  现已内置**保守的方向归因**（`measurement.comparative_attribution: true` 开启，仅对明确句式给方向、含糊则丢弃），
  可作稳健性附录。注意主指标 rating 逐个询问、本就**不存在**比较句/归因问题，construct validity 主要由它承担。
- **归因匹配**：已修复旧版 `"ai"` 子串误命中（explain/remain/…）的 bug，改用词边界匹配；中文先剥离 AI 复合词再判人类指称。
- **模型非独立性**：所测 instruct 模型多有同源血缘（Llama 系/蒸馏自同一基座），属非独立样本，跨模型泛化需谨慎；
  见 `by_family.csv`，lme4 的 `(1|model)` 已部分汇聚，必要时可加 `(1|model_family)`。
- **base 模型不会聊天**：若你真拿到 base 模型，对它要用 completion 风格提示（让它补全数字），否则它会续写而非作答。
- **seed 非万能**：默认关 seed，靠重复次数估方差；这是有意为之，别依赖单次可复现。

---

## 目录结构

```
valence-audit/
  run_all.py            一键入口（mock/list-models/smoke/full）
  analyze_lme4.R        交叉随机效应（论文最终建模）
  config/
    experiment.yaml     运行参数（改这里）
    models.yaml         模型清单（含 base 段）
    dimensions.yaml     8 维度的中英措辞
  src/
    config.py  build_prompts.py  providers.py
    generate.py  measure.py  analyze.py  figures.py
  data/                 运行时产出（raw/ measured/ results/）
  requirements.txt  .env.example  修改说明.md
```
