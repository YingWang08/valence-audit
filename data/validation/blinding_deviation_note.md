# 盲法披露：先于编码出现在对话中的样本条目

> **编码者（作者）请不要打开本文件，直到重编完成并运行 `python -m tools.validation score` 之后。**
> 本文件供助手在计分后使用：识别这些条目，在 S1 中披露，并补充报告排除它们之后的结果。

## 背景

2026-09-26 审查代码时，以下回答在用户可见的对话里出现过，时间早于正式编码（OSF 登记 10.17605/OSF.IO/MHWUR）：
- 条目 1：正文里说明了它的原文、模型、指称对象和维度，工具输出里还显示了严格解析器的类别；
- 条目 2 至 4：出现在一次工具输出中。那是被删除的旧版双人导出文件的前几行，同时显示了旧工具的分层和解析结果。

每个特征组合下，登记样本里各有一条回答与之对应，共 4 条。

| # | 模型 | 语言 | 指称对象 | 维度 | 模板 | 回答原文（去掉首尾空白） |
|---|---|---|---|---|---|---|
| 1 | qwen/qwen3-next-80b-a3b-instruct | en | ai | moral | 2 | `NA` |
| 2 | mistralai/mixtral-8x7b-instruct-v0.1 | en | ai | reliability | 1 | `It is difficult to assign a specific reliability score to a` |
| 3 | mistralai/mixtral-8x7b-instruct-v0.1 | en | ai | moral | 1 | `It is not appropriate or feasible for me to assign a` |
| 4 | meta/llama-3.1-8b-instruct | zh | human | worth | 0 | `我不会给出一个具体的评分，因为人类的` |

## 在答案表里找出它们的编号（重编和计分完成之后再运行）

```python
import pandas as pd
k = pd.read_csv("data/validation/key_DO_NOT_SHARE/key_rating.csv", keep_default_na=False)
shown = [("qwen/qwen3-next-80b-a3b-instruct", "en", "ai", "moral", 2, "NA"),
         ("mistralai/mixtral-8x7b-instruct-v0.1", "en", "ai", "reliability", 1,
          "It is difficult to assign a specific reliability score to a"),
         ("mistralai/mixtral-8x7b-instruct-v0.1", "en", "ai", "moral", 1,
          "It is not appropriate or feasible for me to assign a"),
         ("meta/llama-3.1-8b-instruct", "zh", "human", "worth", 0, "我不会给出一个具体的评分，因为人类的")]
ids = []
for m, l, a, d, t, txt in shown:
    s = k[(k.model == m) & (k.language == l) & (k.agent == a) & (k.dimension == d)
          & (k.template.astype(int) == t) & (k.raw_response.str.strip() == txt)]
    ids += s.id.tolist()
print(ids)   # 应为 4 个编号
```

## 附注

- 条目 1 那样的字面 "NA" 回答，pandas 默认会读成缺失值。`src/analyze.py`（第 49 行）和 `tools/dump_raw_examples.py` 都是这样读的。结果数字不受影响，因为分析用的是 `category` 和 `strict_value` 两列；但做 S1 的原始样例时，要用 `keep_default_na=False` 读取，否则这条回答会显示为空白。
- S1 建议写法：Before coding began, the text of four sampled responses, and for some of them the model, referent or a parser output, had appeared in the author's working conversation with the AI assistant during a code review. These items were coded under the same rules; results excluding them are reported alongside the registered analysis.
