# Results: seven setups with the new prompt

I ran every setup on the same seeded sample of 20 dev questions as in round 1 ([first_results.md](first_results.md)),
without the judge: an answer run and an abstention run each. The only change from round 1 is the refusal instruction:
the prompt no longer dictates the exact refusal sentence, but asks the model to *"say so in one sentence and cite
nothing"*. Model, retrieval, answer rules (version 4) and sample are unchanged. The baseline then ran on all 60 dev
questions with the judge (phase 2a, at the end of this doc). The files are in `results/archive/round_2026-10-24/`.

## Prompt

The exact system prompt of this round (fingerprint `c745ef277954`, stored with all 280 records;
[prompt.py](../src/rag_assistant/prompt.py)):

```text
You are a technical assistant for LangChain and related libraries.
Answer the question using ONLY the sources provided by the user.
- If the sources contain information relevant to the question, answer with what the sources support, even if it
  only covers part of the question, and set insufficient_information to false. Say which part you cannot answer.
- Set insufficient_information to true ONLY if none of the sources is about the question's topic. Then say so in one
  sentence and cite nothing. Never answer from outside knowledge.
- Cite every source you use by its exact doc_id in cited_doc_ids. Only cite doc_ids that appear in the sources.
- Keep the answer concise. Include code from the sources when it helps.
```

The user message, output schema and settings are the same as in round 1: the sources followed by the question;
`{"answer", "cited_doc_ids", "insufficient_information"}`, at most 800 tokens; `llama3.2:3b`, temperature 0, seed 42,
`num_ctx` 8192. A refusal is always shown with the standard sentence *"The knowledge source does not contain enough
information to answer this question."*, whatever the model wrote.

## Results per setup (20-question sample)

| Setup | Ctx Cov | Answered | Correct abst | Cit prec | Time / question |
|---|---:|---:|---:|---:|---:|
| B baseline (k4, 2,000) | 0.254 | 85% | 15% | 0.14 | 43.8 s |
| 1 depth (k2, 4,000) | 0.183 | 85% | **25%** | **0.23** | 38.8 s |
| 2 speed (k3, 1,500) | 0.196 | 85% | 15% | 0.19 | **29.8 s** |
| 3 `identifier_split` (k4, 2,000) | 0.225 | 80% | 20% | 0.13 | 44.3 s |
| 4 more context (k6, 2,000) | **0.271** | 75% | 15% | 0.20 | 67.1 s |
| 5 breadth (k8, 1,000) | **0.271** | **95%** | 5% | 0.14 | 54.6 s |
| 6 less truncation (k4, 3,000) | 0.254 | 85% | 5% | 0.15 | 53.7 s |

- **Answered:** share of the answerable questions answered.
- **Correct abst:** share refused when the context was only non-relevant chunks.
- **Cit prec:** share of citations that FreshStack judged relevant; a lower bound, because 23-41% of the cited chunks
  were never judged.
- **Time:** generation time per question in the answer run.
- **Bold** marks the best value. Only speed differences are clear (95% interval excludes 0); see below.
- **Invalid output:** 5 of 280 outputs (setups 4 and 5 in the answer run; B, 1 and 2 in the abstention run). All were
  cut off at the 800-token limit.

**Per-question comparison with the baseline (`rag eval compare`, setup − baseline, 95% interval; **bold** = clear):**

| Setup | Answered | Correct abst | Ctx Cov | Time (answer run) |
|---|---|---|---|---|
| 1 | +0.00 [−0.15, +0.15] | +0.10 [−0.10, +0.30] | −0.07 [−0.16, 0.00] | −5.0 s [−9.4, +0.2] |
| 2 | +0.00 [−0.20, +0.20] | +0.00 [−0.25, +0.25] | −0.06 [−0.15, 0.00] | **−14.0 s [−16.5, −11.6]** |
| 3 | −0.05 [−0.15, 0.00] | +0.05 [−0.10, +0.20] | −0.03 [−0.14, +0.11] | +0.4 s [−4.7, +6.7] |
| 4 | −0.10 [−0.35, +0.15] | +0.00 [−0.25, +0.25] | +0.02 [0.00, +0.05] | **+23.3 s [+16.3, +31.1]** |
| 5 | +0.10 [−0.10, +0.30] | −0.10 [−0.25, 0.00] | +0.02 [0.00, +0.05] | **+10.8 s [+4.5, +18.5]** |
| 6 | +0.00 [−0.20, +0.20] | −0.10 [−0.25, 0.00] | 0.00 (same chunks) | **+9.9 s [+6.4, +13.1]** |

**Compared with round 1 (baseline, same 20 questions):**

| | Round 1 (exact sentence) | Round 2 (new prompt) |
|---|---:|---:|
| Answered | 20% | **85%** |
| Correct abstention | **70%** | 15% |
| Time per question | 37.2 s | 43.8 s |

## Findings

- **The over-refusal of round 1 is gone.** Every setup answered 75-95% of the questions (round 1: 10-30%).
- **Abstention collapsed instead.** Every setup answered 75-95% of the contexts that cannot answer the question
  (correct abstention 5-25%, round 1: 70-100%). Of those false answers, 93-100% cite one of the non-relevant chunks as
  if it supported the answer.
- **The model still does not judge whether the evidence is sufficient.** The prompt moved the default from refusing
  (round 1) to answering: with either prompt, the model either answers or refuses almost everything, whatever the
  context.
- **Accuracy:**
  - **No setup is clearly better** in answered questions, abstention or citation precision; all differences are within
    the noise of 20 questions.
  - **Setup 1 is best on two quality metrics without a clear difference:** the highest correct abstention (25%) and
    citation precision (0.23, partly because it can cite at most 2 chunks), but 2 answers without a valid citation.
  - **Setups 5 and 6 abstain worst** (5%): they answered 19 of the 20 insufficient contexts. The interval of the
    difference touches 0 but does not exclude it.
- **Evidence:** context coverage is 0.18-0.27, the same as in round 1 (retrieval did not change). Fewer chunks
  (setups 1 and 2) lose some evidence (−0.06 to −0.07, borderline); more chunks (4 and 5) add almost none (+0.02).
- **Speed:** setup 2 is clearly the fastest (−14 s per question, ~30% faster); setup 1 is slightly faster (−5 s, not
  clear). More text costs time: setups 4, 5 and 6 are clearly slower (+10 to +23 s). Answering takes longer than
  refusing: the baseline needs 43.8 s per question, against 37.2 s in round 1.
- **Costs:** all runs are local and free; the cost is compute time, which follows the amount of text in the prompt:
  from 4,500 characters of chunks (setup 2) to 12,000 (setups 4 and 6). No prompt filled the 8,192-token context
  window (no warning in the run logs).

## Winner

**Setup 2 (speed: top-k 3, 1,500 characters) is the best of round 2, by a small margin.**
- **Clearly faster:** −14 s per question (29.8 s against 43.8 s), the only clear difference in this round.
- **Cheaper:** the least text in the prompt (4,500 characters of chunks, against 8,000).
- **No measurable loss:** the same answered rate (85%) and correct abstention (15%) as the baseline, and a slightly
  higher citation precision (0.19 against 0.14, not clear).
- **Against it:** less evidence in its context (Ctx Cov 0.20 against 0.25, borderline).

It is the candidate to compare with the baseline on the full dev set, where the judge can show whether the smaller
context costs answer quality. **Setup 1** is the alternative if abstention and citation precision weigh more than
speed: it scores best on both, but neither difference is clear and it sees the least evidence (Ctx Cov 0.18).

## Limitations

- **Abstention is the weak point.** With this prompt the model answers almost every context, including 75-95% of the
  contexts that cannot answer the question, usually citing a chunk that does not back the answer. No setup changes
  that. → Treat abstention as a limitation of the prompt-based approach, not of the setup. Making refusing depend on
  checkable evidence ("evidence first": refuse in code when the model cannot quote supporting text from the sources)
  is the next lever.
- **No judge in phase 1.** The setup comparison shows whether the model answers, cites and refuses, not whether the
  answers are correct. → Phase 2a on all 60 dev questions, with the judge (below).
- **Small sample.** With 20 questions, one question is 5 points; only the speed differences are clear. → Confirm the
  chosen setup on all 60 dev questions.
- **Citation precision is a lower bound.** 23-41% of the cited chunks were never judged by FreshStack and count as
  not relevant.
- **Long answers are cut off.** 5 of 280 outputs hit the 800-token limit and became invalid, now that the model answers
  instead of refusing. → Raise `max_tokens`, or accept it as a rare failure (2%).
- **Retrieval caps everything.** Context coverage ~0.25 limits every setup alike: on average only a quarter of the key
  facts reach the model. → Better retrieval (dense or hybrid) is likely to matter more than the context
  budget.

## Conclusion

The new prompt fixed the over-refusal of round 1, but moved the problem to the other side: the model now answers
almost everything, including contexts that cannot answer the question. Neither prompt makes it refuse selectively.
Within this round the setups differ clearly only in speed, so the context budget is a speed trade-off, not a quality
one, as far as 20 questions can show.

**Next:**
- Run phase 2 on all 60 dev questions with the judge: the baseline (phase 2a) and setup 2 (phase 2b), and choose
  between them on the deterministic metrics first, the judge metrics second.
- Weak abstention is a limitation; "evidence first" is the most promising fix, but it needs a new
  prompt and answer format.

## Phase 2a: the baseline on all 60 dev questions, with the judge

Only the baseline was run; setup 2 (phase 2b) was not, and the baseline setup was kept.

| Metric | Value |
|---|---:|
| Answered | 77% |
| Correct abstention (insufficient context) | 17% |
| False answers on insufficient context citing a non-relevant chunk | 98% |
| Nugget recall, all questions / over answered questions (judge) | 0.41 / 0.54 |
| Citation precision / citation support (judge) | 0.21 / 0.20 |
| Answers without a valid citation | 2% |
| Invalid output / judge parse failures | 0% / 0% |
| Time per question (generation + judge) | 40 s + 70 s |

- **The sample result holds:** the model answers almost everything, including 82% of the insufficient contexts.
- **The judge is healthy:** no parse failures, and citation support close to citation precision.
- **What happened next:** the prompt turned out to contradict itself ("use ONLY the sources", but refuse "ONLY if none
  of the sources is about the topic"). Round 3 rewrote it; see [third_results.md](third_results.md) and
  [experiments.md](experiments.md).
