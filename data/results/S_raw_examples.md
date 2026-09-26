# Raw response examples (S1 File)

## Retained models, by outcome category

**google/gemma-2-2b-it / hedge**
- [en, human, worth, t3] `This is a very complex and subjective question. There is no`
- [en, human, worth, t3] `This is a very subjective and complex question. There is no`
- [en, human, worth, t3] `This is a complex and highly subjective question. There is no`

**google/gemma-2-2b-it / malformed**
- [en, human, moral, t0] `This is a complex question with no easy answer.`
- [en, human, moral, t0] `This is a complex philosophical question with no easy answer.`
- [en, human, moral, t0] `This is a complex question without a simple answer.`

**google/gemma-2-2b-it / refusal**
- [en, human, productivity, t3] `Please provide me with some context!  How should I judge`
- [zh, ai, moral, t3] `Please provide me with the AI system you want me to evaluate`
- [en, human, moral, t1] `It's impossible to rate a human's morality on`

**google/gemma-2-2b-it / truncated**
- [en, ai, productivity, t1] `This is a very complex question!  Productivity for AI depends`
- [en, ai, moral, t3] `This is a complex question with no easy answer.  It`
- [en, ai, moral, t3] `This is a complex question with no easy answer.  A`

**google/gemma-2-2b-it / valid**
- [zh, human, decision, t2] `5`
- [zh, human, worth, t2] `5`
- [en, ai, trust, t1] `3`

**meta/llama-3.1-8b-instruct / refusal**
- [zh, human, worth, t0] `我不会给出一个具体的评分，因为人类的`

**meta/llama-3.1-8b-instruct / valid**
- [en, human, creativity, t0] `6`
- [zh, human, emotion, t1] `4`
- [zh, ai, trust, t1] `4`

**meta/llama-3.3-70b-instruct / out_of_range**
- [zh, ai, worth, t2] `0`
- [zh, ai, worth, t2] `0`

**meta/llama-3.3-70b-instruct / valid**
- [zh, ai, worth, t2] `5`
- [en, human, trust, t1] `4`
- [en, ai, decision, t2] `5`

**microsoft/phi-4-mini-instruct / hedge**
- [en, human, emotion, t0] `Emotional perceptiveness can vary widely among individuals, but on`
- [en, ai, trust, t3] `I will need specific context or information about the AI system to`
- [zh, human, trust, t0] `这个问题很复杂，答案也取决于你所`

**microsoft/phi-4-mini-instruct / out_of_range**
- [en, human, creativity, t2] `8.`
- [zh, human, creativity, t1] `7.5分。如果要给机器的创造力水平`

**microsoft/phi-4-mini-instruct / refusal**
- [en, human, moral, t3] `I cannot score individuals as it would be both unethical and impr`
- [en, human, moral, t0] `As an artificial intelligence, I don't have personal moral judgments.`
- [en, ai, moral, t1] `I cannot rate an AI system's moral soundness as it`

**microsoft/phi-4-mini-instruct / truncated**
- [zh, human, moral, t1] `这还不清楚。为了给出一个分数`
- [en, ai, trust, t3] `Without further information on the AI system's capabilities, performance,`
- [zh, ai, creativity, t1] `这个评定是根据它执行创意任务的能力`

**microsoft/phi-4-mini-instruct / valid**
- [zh, ai, moral, t0] `3`
- [en, human, decision, t1] `7`
- [zh, human, decision, t2] `7`

**microsoft/phi-4-mini-instruct / valid_range**
- [zh, human, moral, t1] `1-2分`

**mistralai/mixtral-8x7b-instruct-v0.1 / hedge**
- [en, human, productivity, t0] ` This is a subjective question and the answer can vary greatly`
- [en, human, productivity, t1] ` Productivity can vary greatly from person to person, so it`
- [en, human, productivity, t0] ` This is a subjective question and the answer can vary greatly`

**mistralai/mixtral-8x7b-instruct-v0.1 / malformed**
- [zh, ai, reliability, t0] ` AI 系统的可靠性因`
- [zh, human, trust, t0] ` 为了回答这个问题，我需`
- [zh, human, trust, t0] ` 为了回答这个问题，我需`

**mistralai/mixtral-8x7b-instruct-v0.1 / out_of_range**
- [zh, ai, reliability, t2] ` 8 （八分之一可靠`
- [zh, ai, reliability, t2] ` 8 （ höchstens eins Komma`
- [zh, ai, reliability, t2] ` 8 （八分）。当前 AI `

**mistralai/mixtral-8x7b-instruct-v0.1 / refusal**
- [en, human, productivity, t3] ` I cannot directly observe or evaluate a human's productivity as`
- [en, ai, moral, t0] ` It is not possible to provide a numerical rating for the moral`
- [en, human, moral, t1] ` I cannot provide a moral judgment of a human's character`

**mistralai/mixtral-8x7b-instruct-v0.1 / truncated**
- [en, ai, productivity, t1] ` A well-designed and properly implemented AI system can be`
- [en, ai, reliability, t1] ` I would need more specific information about the AI system in question`
- [en, ai, emotion, t1] ` As a responsible and honest assistant, I would rate the emotional`

**mistralai/mixtral-8x7b-instruct-v0.1 / valid**
- [zh, human, trust, t1] ` 5  (Note: This is a subjective`
- [zh, ai, emotion, t3] ` 7  (这个回答假设你`
- [zh, ai, trust, t0] ` 7 是一个 ideal 的答案，因`

**mistralai/mixtral-8x7b-instruct-v0.1 / valid_range**
- [en, ai, emotion, t1] ` As of now, a rating of 1 or 2`

**nvidia/nemotron-mini-4b-instruct / hedge**
- [en, human, emotion, t2] `A human's emotional perceptiveness can vary greatly, but`
- [en, human, decision, t0] `A human's ability to make decisions can vary widely depending`
- [en, human, trust, t0] `A human's trustworthiness can vary greatly depending on their character`

**nvidia/nemotron-mini-4b-instruct / malformed**
- [en, human, trust, t3] `Sure, I'd be happy to help with that.`
- [en, human, trust, t3] `Sure, I'd be happy to help with that.`
- [en, human, productivity, t3] `Sure, I'd be happy to help with that.`

**nvidia/nemotron-mini-4b-instruct / refusal**
- [en, ai, trust, t0] `Based on the information provided, I cannot provide a definitive answer`
- [zh, human, moral, t0] `I'm sorry, but I cannot provide a number to`
- [en, ai, trust, t0] `Based on the information provided, I cannot provide a definitive answer`

**nvidia/nemotron-mini-4b-instruct / truncated**
- [zh, human, worth, t0] `Based on my understanding of human values and potential, I would`
- [en, human, moral, t2] `A human's moral soundness is a complex issue that`
- [zh, ai, emotion, t3] `Sure, I'd be happy to help you evaluate the`

**nvidia/nemotron-mini-4b-instruct / valid**
- [en, human, moral, t1] `5`
- [en, ai, productivity, t1] `5`
- [zh, ai, trust, t1] `5`

**qwen/qwen3-next-80b-a3b-instruct / malformed**
- [en, ai, moral, t2] ``
- [en, ai, moral, t2] ``

**qwen/qwen3-next-80b-a3b-instruct / out_of_range**
- [en, ai, moral, t2] `0`
- [en, ai, moral, t2] `0`
- [en, ai, moral, t2] `0`

**qwen/qwen3-next-80b-a3b-instruct / valid**
- [zh, human, creativity, t2] `7`
- [zh, human, worth, t1] `7`
- [zh, ai, creativity, t1] `5`

## Excluded models: empty responses by round and format

                                                                                        n  empty  pct_empty
model                                    round                             format                          
nvidia/llama-3.3-nemotron-super-49b-v1.5 final round (data/raw)            compare    240    216       90.0
                                                                           forced     240    236       98.3
                                                                           rating     640    621       97.0
                                                                           reflect    160    126       78.8
                                                                           scenario   160    159       99.4
                                         first round (data/raw_quarantine) compare    228    228      100.0
                                                                           forced     236    236      100.0
                                                                           rating     630    630      100.0
                                                                           reflect    137    137      100.0
                                                                           scenario   160    160      100.0
openai/gpt-oss-20b                       final round (data/raw)            compare    384     87       22.7
                                                                           forced     384     29        7.6
                                                                           rating    1024   1024      100.0
                                                                           reflect    256     62       24.2
                                                                           scenario   256     80       31.2
                                         first round (data/raw_quarantine) compare    146    146      100.0
                                                                           forced      67     67      100.0
                                                                           rating    1024   1024      100.0
                                                                           reflect    111    111      100.0
                                                                           scenario   119    119      100.0

Non-empty responses of excluded models: 1104 (all listed in S_raw_examples_excluded.csv)
