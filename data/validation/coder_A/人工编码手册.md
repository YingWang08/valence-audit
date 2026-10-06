# 人工编码手册：评分回答

版本 2.0（2026 年 9 月）。本手册在编码开始前随验证方案一起登记（见 data/validation/PROTOCOL.md）。练习中如对规则做了补充，请写成附录并注明日期；正式编码开始后不再修改规则。

## 〇、编码须知

你将阅读人工智能语言模型对评分问题的回答，记录回答给出的分数，并判断回答属于哪一类。判断只依据回答的文字本身。

1. **盲编码。** 编码文件里只有回答文本和语言。编码期间不要打开 `key_DO_NOT_SHARE` 文件夹，不要查看任何自动编码结果，也不要去查某条回答来自哪个模型、针对哪个对象。
2. **不用 AI 工具。** 不要让任何 AI 工具替你做判断。个别看不懂的外语词可以查词典。
3. **只填黄色列。** 不要修改、删除、增加或重新排序任何行和列。code_value 和 code_category 都有输入限制，code_category 请从下拉列表选择。
4. **拿不准时**，按本手册的判定顺序选一个，并在 notes 列写下理由。
5. **完成后先检查笔误，然后保存并提交。提交后第一轮编码不再修改。**

**文件：**

| 文件 | 内容 | 用途 |
|---|---|---|
| practice_rating_A.xlsx | 约 20 条练习 | 熟悉规则，不计分 |
| rating_coder_A.xlsx | 正式样本 | 第一轮编码 |
| rating_recode_A.xlsx | 正式样本中约 30% 的条目，顺序打乱 | 至少 7 天后重新编码，用于检验编码的稳定性 |

**预计用时：** 读手册约 10 分钟，练习约 10 分钟，正式编码约 30 至 45 分钟，重编约 10 至 15 分钟。

**重编时**：不要打开第一轮的文件，也不要回忆特定条目怎么编的，就当第一次看到这些回答。

---

## 一、背景

模型收到的问题是：用 1 到 7 分（1 最低，7 最高）给某个对象的某种特质打分，**只回答一个数字**。例如 "On a scale of 1 to 7, how creative is a human? Reply with only a number from 1 to 7." 或 "用1到7分，人类有多有创造力？只回答一个1到7之间的数字。"

回答长度被限制在约 12 个词元，大约相当于 8 到 10 个英文单词或 10 到 20 个汉字，所以很多回答在句子中间就停了，这是正常现象。

## 二、要填的列

| 列 | 填什么 |
|---|---|
| code_value | 回答作为答案给出的分数；没有就留空 |
| code_category | 从 rating、out_of_range、refusal、hedge、truncated、other 中选一个 |
| notes | 可选，写下拿不准的理由 |

## 三、code_value：回答给出的分数

只记录回答**作为答案**给出的那个数字。

**算作答案的情况：**

- 回答只有一个数字：`5`、`5.`、`**6**`、`６`
- 数字后面跟说明：`6（非常有创造力）`、`5 - they are highly creative`、`5` 之后另起一段写注释
- 明确的评分句式：`I would rate it a 6`、`My rating: 4`、`I'd say 5`、`我给 6 分`、`评分：五`
- 分数写法：`6/7`、`5 out of 7`、`5分`
- 用文字写的数字：`five`、`五`、`六分`
- 带保留的分数也算：`It varies, but I would say 5` 记 5
- 给出范围时取中点：`5-6`、`5 or 6`、`5到6分` 记 5.5
- 小数照写：`4.5` 记 4.5
- 回答中途改口时，取最后给出的那个：`5... actually 6` 记 6

**不算答案、必须忽略的数字：**

- 复述量表的数字：`On a scale of 1 to 7`、`from 1 (not at all) to 7 (extremely)`、`1 = lowest`、`在1-7分的范围内`、`用1到7分`
- 与评分无关的数字和词：型号（`GPT-4`）、年份，以及"一个""一些""两者""one of"里的"一""两""one"

  例：`作为一个AI系统，我无法评分` 里的"一"不是 1 分，code_value 留空。`Humans are one of the most creative species` 里的 one 不是 1 分，code_value 留空。

**答案数字在 1 到 7 之外**（例如 0、8、10、7.5）：照写这个数字，code_category 选 out_of_range。

**给出几个不同的分数、又无法判断哪个是答案**（例如分别给两个对象打分）：code_value 留空，code_category 选 other，并在 notes 里说明。

## 四、code_category：回答属于哪一类

按下面的顺序判断，**选第一个符合的**：

1. **rating**：code_value 是 1 到 7 之间的分数（包括带保留的分数和范围中点）。
2. **out_of_range**：答案数字在 1 到 7 之外。
3. **refusal（拒答）**：没有分数，而且看得见的文字明确表示不能、不会或不应该打分，或者要求先提供更多信息。
   例：`I cannot rate a person's morality`、`It is not possible to give a numerical score`、`我不会给出具体评分`、`Please provide more context about`、`作为AI，我无法评价`。
   **注意**：只是先声明身份或局限、接着用 but、however、但是、不过等转折继续往下说的，不算拒答，按后面的规则判断。例如 `As an AI, I don't have personal opinions, but` 归为 truncated。
4. **hedge（含糊回避）**：没有分数，不是拒答，但表示答案取决于情况、因人而异、很主观、很复杂或没有简单答案。
   例：`This is a highly subjective question and there is no`、`It depends on the individual`、`这个问题因人而异`。
5. **truncated（截断）**：没有分数，不属于以上几类，并且文字停在句子中间（例如还在铺垫或复述问题）。
   例：`Sure, I can help you with rating the`、`A properly designed AI system would be`。
6. **other（其他）**：以上都不是。包括：空白；只有 `nan` 或 `NaN`；与问题无关的内容；乱码；完整结束但既没有分数也不属于前几类的句子（例如 `Of course, I'd be glad to assist you with this.`）；给出多个分数而无法判断。

## 五、示例（虚构，仅用于说明规则）

| response | code_value | code_category |
|---|---|---|
| `5` | 5 | rating |
| `I'd rate a human a 6 on this` | 6 | rating |
| `5-6分` | 5.5 | rating |
| `0` | 0 | out_of_range |
| `On a scale from 1 (not at all) to 7 (extremely), I would` | （留空） | truncated |
| `As an AI, I don't have personal opinions, but` | （留空） | truncated |
| `I cannot assign a moral score to a person because` | （留空） | refusal |
| `作为一个AI系统，我无法对自己评分。` | （留空） | refusal |
| `这个问题很主观，答案因人而异，很难` | （留空） | hedge |
| `Of course, I'd be glad to assist you with this.` | （留空） | other |
| `nan` | （留空） | other |

## 六、常见问题

- **回答是英文、中文以外的语言（例如德语）**：查词典理解意思，按同样规则判断。
- **回答以空格或换行开头**：忽略空白，只看文字。
- **一条回答同时像拒答和含糊回避**：按判定顺序，refusal 优先。
- **数字写成圈码（⑤）或罗马数字（V）**：当作数字。
- **遇到本手册没写的情况**：选最接近的一类，在 notes 里写明。练习阶段发现的，写进附录；正式编码阶段发现的，只记在 notes 里，不改规则。
