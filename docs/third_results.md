# Results: the baseline with a stricter grounding prompt

Round 2 ([second_results.md](second_results.md)) fixed the over-refusal of round 1, but the model then answered almost
every context, including 75-95% of the contexts that cannot answer the question. Its prompt also contradicted itself:
"use ONLY the sources", but refuse "ONLY if none of the sources is about the question's topic". This round tests one
change: a prompt that makes grounding explicit and asks to refuse whenever the sources do not contain the answer.

I ran only the baseline (top-k 4, 2,000 characters, `word_stem`) on the same seeded sample of 20 dev questions as rounds
1 and 2, without the judge: an answer run and an abstention run. Model, retrieval, answer rules (version 4), schema and
sample are unchanged, so any difference comes from the prompt wording. Round 2 is archived in
`results/archive/round_2026-10-24/`; its baseline sample runs were kept in `results/` for the paired comparison.

## Prompt

The exact system prompt of this round (fingerprint `f2d2db446790`; [prompt.py](../src/rag_assistant/prompt.py)):

```text
You are a technical assistant for LangChain and related libraries.
Answer ONLY with information that appears in the sources provided by the user. Do not add facts, APIs or code from
your own knowledge.
- If the sources answer the question, or part of it, answer that part, set insufficient_information to false and say
  which part the sources do not cover.
- If the sources do not contain the answer, set insufficient_information to true, say so in one sentence and cite
  nothing, even if the sources are about the same topic.
- Every statement in the answer must be supported by a source you cite by its exact doc_id in cited_doc_ids. Only cite
  doc_ids that appear in the sources.
- Keep the answer concise. Include code from the sources when it helps.
```

What changed from round 2 (`c745ef277954`):
- **"No outside knowledge" is a general rule**, not part of the refusal bullet.
- **The refusal threshold** is "the sources do not contain the answer", explicitly *"even if the sources are about the
  same topic"*, instead of "none of the sources is about the topic".
- **Citations are tied to statements:** every statement must be supported by a cited source.

The user message, output schema (answer first, flag last) and settings are unchanged: at most 800 tokens,
`llama3.2:3b`, temperature 0, seed 42, `num_ctx` 8192.

## Results (baseline, 20-question sample)

| Metric | Round 2 | Round 3 |
|---|---:|---:|
| **Correct abstention (insufficient context)** | 15% | **60%** |
| False answers on insufficient context citing a non-relevant chunk | 100% | 100% |
| **Answered** | **85%** | 55% |
| Context coverage (same retrieval) | 0.254 | 0.254 |
| Citation precision (vs FreshStack, lower bound) | 0.14 (42 citations) | 0.08 (24 citations) |
| Answers without a valid citation | 0% | 0% |
| Invalid output (answer run / abstention run) | 0% / 5% | 0% / 0% |
| Generation time per question (answer run / abstention run) | 43.8 s / 47.2 s | 41.5 s / 41.1 s |

Refusals are always shown with the standard sentence, whatever the model wrote.

**Per-question comparison (`rag eval compare`, round 3 − round 2, 95% interval; bold = clear):**

| Metric | Difference | Questions higher / lower / same |
|---|---|---|
| Correct abstention | **+0.45 [+0.25, +0.65]** | 9 / 0 / 11 |
| False answer (insufficient context) | **−0.40 [−0.60, −0.20]** | 0 / 8 / 12 |
| Answered (real context) | **−0.30 [−0.55, −0.05]** | 1 / 7 / 12 |
| Citation precision (10 questions answered in both) | −0.02 [−0.27, +0.18] | 2 / 1 / 7 |
| Generation time, answer run | −2.3 s [−6.6, +1.6] | 10 / 10 / 0 |
| Generation time, abstention run | **−6.2 s [−14.0, −0.2]** | 6 / 14 / 0 |

## Findings

- **The wording works in the intended direction, and clearly.** Correct abstention rose from 15% to 60%: 9 questions
  improved and none got worse. False answers on insufficient context halved.
- **The price is fewer answers.** Answered dropped from 85% to 55% (7 questions fewer, 1 more). With context coverage
  at 0.25, many real contexts contain little or none of the answer, so part of this drop is correct behaviour,
  although the answer evaluation counts every refusal as a false abstention. Whether the lost answers were correct is
  not measured; this phase has no judge.
- **It is still far from reliable.** 40% of the insufficient contexts are still answered, and every such answer cites
  a non-relevant chunk.
- **Citations:** precision stays low (0.08-0.14) and its difference is not clear; with 24 citations in round 3 it is
  barely measurable.
- **Speed:** slightly faster, because refusals are short (−6 s per question in the abstention run, clear).

## Decision on the sample

**The new prompt is the better candidate.** Correct abstention rose clearly (+0.45), and the model still answers more
than half of the questions, well above round 1's 20%. The cost, a clear drop in answered questions, must be checked
with the judge on more questions.

The rule for the full dev run, set before it: keep the new prompt if correct abstention stays clearly higher without
a clear loss of nugget recall; otherwise return to the round-2 prompt. The rule was first formulated for the questions
whose context holds evidence; that split is not computed by the repository's evaluation, so the check below uses
nugget recall over all questions, which is the stricter test.

## Phase 2a: the full dev set (60 questions), with the judge

The baseline with the new prompt on all 60 dev questions, with the judge (`llama3`, version 5), compared per question
with the round-2 phase 2a run (archived in `results/archive/round_2026-10-24/`; compared with `rag eval compare` as
described in [how_to_run.md](how_to_run.md), step 5). Same model, retrieval, answer rules and judge; only the prompt
differs.

| Metric | Round 2 | Round 3 | Round 3 − round 2 (95% interval) |
|---|---:|---:|---|
| **Correct abstention (insufficient context)** | 17% | **50%** | **+0.33 [+0.20, +0.47]**, 22 better / 2 worse |
| False answers on insufficient context citing a non-relevant chunk | 98% | 86% | |
| Answered | **77%** | 52% | **−0.25 [−0.38, −0.12]**, 3 more / 18 fewer |
| **Nugget recall, all questions (judge)** | **0.41** | 0.32 | −0.10 [−0.20, +0.00], not clear |
| Nugget recall over answered questions (judge) | 0.54 | **0.61** | |
| Citation precision (vs FreshStack, lower bound) | **0.21** (104 citations) | 0.15 (52) | +0.07 [−0.06, +0.17] per question |
| Citation support (judge, verified quote) | 0.20 | **0.23** | |
| Unjudged share of the cited chunks | 0.32 | 0.33 | |
| Answers without a valid citation | **2%** | 16% | |
| Answers with a dropped / repaired citation | 2% / 10% | 5% / 17% | |
| Invalid output / judge parse failures | 0% / 0% | 0% / 0% | |
| Time per question (generation + judge) | 40 s + 70 s | 39 s + 52 s | |

**What it shows:**
- **The sample result holds.** Correct abstention rose clearly, from 17% to 50% (sample: 15% to 60%), and 22 questions
  improved against 2 that got worse.
- **The decision rule is met, narrowly.** Nugget recall over all questions dropped by 0.10, but the 95% interval just
  includes 0 ([−0.20, +0.00]), so the loss is not clear. It is borderline: grounding costs some answers the judge would
  have credited.
- **The answers that remain are slightly better** (nugget recall over answered questions 0.54 to 0.61; citation
  support 0.20 to 0.23), but neither is a clear difference.
- **A new side effect: uncited answers.** 16% of the answers cite nothing, against 2% in round 2, although the prompt
  now says every statement must be cited. Dropped and repaired citations also rose slightly.
- **Still far from reliable.** Half of the insufficient contexts are still answered, and 86% of those answers cite a
  non-relevant chunk.
- **Judge health is fine:** no parse failures, and citation support (0.23) is not below half of citation precision
  (0.15).

## Decision

**Keep the round-3 prompt (`f2d2db446790`) as the final prompt.** It meets the rule set before the run: correct
abstention is clearly higher (17% to 50%), and the drop in nugget recall is not clear, although borderline. It trades
about a quarter of the answers for refusing three times as often when the context cannot answer the question.

**The final configuration:** `llama3.2:3b` (temperature 0, seed 42), BM25 `word_stem` on the full question, top-k 4,
2,000 characters per chunk, prompt `f2d2db446790`, answer rules version 4, judge `llama3` version 5.

This configuration was then run once on the test split; see the next section.

## Test: the final numbers (143 held-out questions, run once)

The frozen configuration on the 143 test questions, run once, with nothing changed before or after. The stored run
configurations of dev and test are identical (prompt, answer rules, models with digests, judge version, settings).

**Retrieval** (BM25 `word_stem`, full question; no LLM):

| Metric | Dev (60) | Test (143) |
|---|---:|---:|
| Cov@20 (share of key facts in the top 20) | 0.527 | 0.533 |
| R@10 / R@100 | 0.167 / 0.406 | 0.175 / 0.371 |
| nDCG@10 / MRR@10 | 0.186 / 0.364 | 0.218 / 0.400 |
| Unjudged share of the top 10 (recall is a lower bound) | 0.45 | 0.43 |

**Answers, citations and abstention:**

| Metric | Dev (60) | Test (143) |
|---|---:|---:|
| **Correct abstention (insufficient context)** | 50% | **46%** |
| **Answered** | 52% | **58%** |
| Context coverage (key facts in the 4 chunks) | 0.26 | 0.32 |
| **Nugget recall over answered questions (judge)** | 0.61 | **0.48** |
| Nugget recall, all questions | 0.32 | 0.28 |
| **Citation precision (vs FreshStack, lower bound)** | 0.15 (52 citations) | **0.28** (147) |
| **Citation support (judge, verified quote)** | 0.23 | **0.37** |
| Unjudged share of the cited chunks | 0.33 | 0.22 |
| Answers without a valid citation | 16% | 18% |
| False answers on insufficient context citing a non-relevant chunk | 86% | 84% |
| Invalid output (answer run / abstention run) | 0% / 2% | 0% / 0.7% |
| Time per question (generation + judge) | 39 s + 52 s | 38 s + 56 s |

**What the test shows:**
- **Abstention generalizes.** Correct abstention is 46% on test against 50% on dev, and about three times the round-2
  prompt's value on dev (17%).
- **Answer quality is lower than on dev.** Over the answered questions, the judge finds 48% of the key facts on test
  against 61% on dev. Part of this is the expected optimism of numbers measured on the set the prompt was chosen on.
- **Citations are better on test.** Precision 0.28 and support 0.37 (dev 0.15 and 0.23), in line with more evidence in
  the context (coverage 0.32 against 0.26). Both are rough: precision counts unjudged chunks as not relevant, and the
  judge is lenient.
- **Guessing is reduced, not solved.** The model still answers 54% of the contexts that cannot answer the question, and
  84% of those answers cite a non-relevant chunk.
- **Uncited answers stay at 18%**, the side effect of the stricter prompt seen on dev.
- **Retrieval is stable** (Cov@20 0.53 on both splits) and remains the cap: about half of the key facts are not in
  BM25's top 20, and only about a third reach the model.
- **The system ran cleanly:** no invalid answer output, one abstention output cut off at the 800-token limit, no
  prompt that filled the context window, no judge parse failures (run logs). The judge health check passes (citation
  support 0.37 against precision 0.28).

**Examples from the test run** (the answers as the model wrote them, shortened):

- **Good: answered from evidence, with a supporting citation.** *"Langchain Pandas dataframe agent answering questions
  through google search"* (76977775). The answer combines the `person_name_search` and `python_repl` tools from the
  csv-agent template and cites that template (`langchain/templates/csv-agent/csv_agent/agent.py_0_3058`). The judge
  credits all 5 key facts and finds a verified supporting quote; FreshStack also judges the chunk relevant.
- **Good: refuses the insufficient context, answers the real one.** *"How can I limit sourceDocs in langchain +
  pinecone response in JS?"* (76473726). With non-relevant chunks only, it gives the standard refusal. With the real
  context it answers, but only with `returnSourceDocuments`: the judge still credits 2 of 3 key facts to this one-word
  answer, an example of its leniency.
- **Poor: a guess on an insufficient context.** *"What does `pip install unstructured[local-inference]` do?"*
  (75927945). Given only non-relevant chunks, it explains that the feature means "the model is trained on a specific
  dataset" and cites a documentation-setup page that says nothing of the kind.
- **Poor: refuses although the context holds evidence.** *"Can't set custom OpenAI model with langchain on nodejs"*
  (75952371): two thirds of the key facts are in the context (coverage 0.67), and the model refuses.
- **Poor: a plausible answer without evidence.** *"Databricks Dolly LLM: empty result when using LangChain with
  context"* (76124316). No relevant chunk in the context; the model invents an explanation about how `predict`
  handles multiline strings, cites an unrelated few-shot notebook, and the judge credits none of the key facts.

## Limitations

- **The prompt was chosen on dev.** The 20-question sample was used for the third time, and the full dev set confirmed
  the choice. The test split, run once, gives the unbiased number; its answer quality is lower than on dev.
- **The context budget was compared with the round-2 prompt only.** With the round-3 prompt only the baseline (top-k 4,
  2,000 characters) was run, so an interaction between the prompt and the context size is untested.
- **60 dev questions.** The drop in overall nugget recall is sizeable but not clear.
- **The judge is a small local model.** It was lenient in round 1; nugget recall and citation support are upper
  estimates.
- **"Evidence" means FreshStack labels.** Unjudged chunks count as not relevant, so citation precision and context
  coverage are lower bounds.
- **Uncited answers.** 16-18% of the answers cite nothing. Enforcing citations in code would turn these into refusals,
  including answers the judge credits.
- **Still a 3B model deciding in one pass.** The flag is still generated after the answer (the schema order did not
  change); putting it first is the next untested lever.

## Conclusion

Making the grounding rule explicit, and moving the refusal threshold from "about the topic" to "contains the answer",
is the first change that made the model refuse much more often when it should. On all 60 dev questions, correct
abstention rose from 17% to 50%, without a clear loss of nugget recall. The price is a quarter fewer answers, some of
which the judge would have credited, and more answers that forget to cite. Abstention is still the weak point.

The single test run confirms this. On 143 held-out questions the final system answers 58% of the questions, refuses
46% of the contexts that cannot answer them, and cites a relevant or supporting chunk more often than on dev
(precision 0.28, support 0.37). Its answers contain about half of the key facts (nugget recall 0.48 over answered
questions), lower than on dev. The limits are the same on both splits: BM25 retrieval, which leaves about half of the
key facts outside its top 20, and a 3B model that still guesses on about half of the insufficient contexts.
