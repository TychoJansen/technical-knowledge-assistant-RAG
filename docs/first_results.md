# Results: seven setups with the exact-refusal prompt

I ran every setup on the same seeded sample of 20 dev questions, without the judge: an answer
run and an abstention run each. The baseline then ran on all 60 dev questions with the judge. In this round the prompt
told the model, when refusing, to *"answer exactly 'The knowledge source does not contain enough information to answer
this question.'"*. The files are in `results/archive/round_2026-10-23/`.

## Prompt

### Assistant

The exact prompt of this round:
[prompt.py](../src/rag_assistant/prompt.py).

System prompt:

```text
You are a technical assistant for LangChain and related libraries.
Answer the question using ONLY the sources provided by the user.
- If the sources contain information relevant to the question, answer with what the sources support, even if it
  only covers part of the question, and set insufficient_information to false. Say which part you cannot answer.
- Set insufficient_information to true ONLY if none of the sources is about the question's topic. Then answer exactly
  "The knowledge source does not contain enough information to answer this question." and cite nothing. Never answer from outside knowledge.
- Cite every source you use by its exact doc_id in cited_doc_ids. Only cite doc_ids that appear in the sources.
- Keep the answer concise. Include code from the sources when it helps.
```

Output schema: `{"answer": string, "cited_doc_ids": [string], "insufficient_information": boolean}`, all required, at
most 800 tokens. `llama3.2:3b`, temperature 0, seed 42, `num_ctx` 8192.

### Judge

The judge (`llama3`, judge version 5, [judge.py](../src/rag_assistant/evaluation/judge.py)) made two kinds of
calls. These are the exact prompts; the judge prompts did not change after this round.

**Nugget judge**: one call per answered question, for all its nuggets.

System prompt:

```text
You are a strict grader of answers to technical questions.
You get an answer and a list of facts labelled fact_1, fact_2, and so on. For each fact, decide whether the answer
states it: the same meaning, possibly in other words or with equivalent code. A fact counts only if the answer clearly
contains it. Give a verdict for every fact: true if the answer contains it, false if not. Do not use outside knowledge
and do not judge whether the answer is correct, only whether it contains the fact.
```
Output schema: one required boolean per fact (`fact_1`, `fact_2`, ...), at most 400 tokens.

**Citation judge**: one call per valid citation of an answered question.

System prompt:

```text
You are a strict grader of citations.
You get a question, an answer and one source the answer cites. Decide whether the source supports the answer: it must
contain information that backs up at least one concrete claim or code example in the answer.
- If it does, set "supported" to true and copy into "quote" the sentence or code line from the SOURCE that backs up the
  answer, exactly as it is written in the source.
- If it does not (the source is about something else, or only shares some keywords with the answer), set "supported" to
  false and "quote" to an empty string.
```
Output schema: `{"supported": boolean, "quote": string}`, both required, at most 300 tokens. A citation counts as
supported only if the judge says yes **and** its quote occurs in the chunk text: at least 80% of the quote as one
contiguous block, after normalising case and whitespace, and at least 15 characters long. Answers shorter than 20
characters, or that only repeat the title, get no judge call: no nuggets, no supported citations.

## Results per setup (20-question sample)

| Setup | Ctx Cov | Answered | Correct abst | Cit prec | Time / question |
|---|---:|---:|---:|---:|---:|
| B baseline (k4, 2,000) | 0.254 | 20% | 70% | 0.00 | 37.2 s |
| 1 depth (k2, 4,000) | 0.183 | **30%** | 70% | 0.00 | **32.5 s** |
| 2 speed (k3, 1,500) | 0.196 | 10% | 85% | 0.00 | **22.6 s** |
| 3 `identifier_split` (k4, 2,000) | 0.225 | 15% | 85% | 0.00 | **34.2 s** |
| 4 more context (k6, 2,000) | 0.271 | 25% | 85% | 0.07 | 49.6 s |
| 5 breadth (k8, 1,000) | 0.271 | 25% | 70% | 0.05 | 37.7 s |
| 6 less truncation (k4, 3,000) | 0.254 | 15% | **100%** | 0.00 | 45.0 s |

- **Answered:** share of the answerable questions answered.
- **Correct abst:** share refused when the context was only non-relevant chunks.
- **Time:** generation time per question in the answer run.
- **Bold** marks the best value. Speed differences are clear (95% interval excludes 0); the others are not.
- No setup produced invalid output.

## Findings

- **Over-refusal dominates everything.** Every setup answered only 10-30% of the questions, although all are
  answerable from the corpus.
- **Refusals do not depend on the context.** The baseline refused the real context (75%) more often than the
  insufficient one (63%). The exact sentence acted as a shortcut: the model repeatedly wrote it while its own flag said
  the information was sufficient (the "refused in words" warnings in the run logs).
- **Accuracy:**
  - **No setup is clearly better** in answered questions or abstention; all differences are within the noise of 20
    questions.
  - **High correct abstention means little here.** Setup 6 refused every insufficient context, but also 85% of the
    answerable questions.
  - **When the model does answer, it always cites,** but correctness is mixed: the judge credits 65% of the key facts,
    yet also gives full credit to wrong answers.
- **Evidence:** context coverage stays at 0.18-0.27 in every setup. More or longer chunks barely add evidence; the limit
  is what BM25 finds. Fewer chunks (setups 1 and 2) lose some.
- **Speed:** setup 2 is clearly the fastest (−15 s per question); setups 1 and 3 are slightly faster (−5 and −3 s).
  More text costs time: setups 4 and 6 are 8-12 s slower.
- **Costs:** The judge roughly triples the time of a full run (33 s generation + 68 s judging per question).

## Winner

**Setup 1 (depth: top-k 2, 4,000 characters) is the best of round 1, by a small margin.**
- **Most answered:** 30%, the most of all setups.
- **Same abstention as the baseline:** 70%.
- **Clearly faster:** −4.8 s per question.
- **Cheaper:** fewer chunks in the prompt, so less compute.
- **Against it:** the least evidence in its context (Ctx Cov 0.18), and none of its quality advantages is clear on 20
  questions.

It is the candidate to compare with the baseline on the full dev set. **Setup 2** is the choice only if speed matters
most: it is the fastest, but it answered the fewest questions. Because the over-refusal swamps the setup differences
(see Limitations), only the baseline runs on the full dev set in this round; the winner is not taken further.

**The baseline on all 60 dev questions, with the judge (phase 2a):**

| Metric | Value |
|---|---:|
| Answered | 25% (15 of 60) |
| Correct abstention (insufficient context) | 63% |
| Nugget recall over answered questions (judge) | 0.649 |
| Citation precision / citation support (judge) | 0.162 / 0.297 |
| Time per question | 33 s generation + 68 s judging |

## Limitations

- **Over-refusal hides the setup differences.** Every setup refused most answerable questions. With so few answers, this round cannot tell which context budget is best. The likely cause is the
  refusal instruction, not the setups: the effect is the same in every setup, and the model repeatedly wrote the exact
  refusal sentence while its own flag said the information was sufficient. → Try a refusal instruction without the
  exact sentence, and rerun the setups with it.
- **The winner may not hold.** Setup 1 won while every setup over-refused. If the model answers more under another
  prompt, the ranking may change. → Choose the setup only once the over-refusal is resolved.
- **Refusals do not follow the evidence.** The baseline refused the real context (75%) about as often as the
  insufficient one (63%), so whether it refuses does not depend on whether the context holds evidence. Another wording
  may change how often the model refuses without making it refuse at the right moments. → Consider making refusing
  depend on checkable evidence: refuse in code when the model cannot quote supporting text from the sources.
- **Small sample.** With 20 questions, one question is 5 points; only the speed differences are clear. → Confirm the
  chosen setup on all 60 dev questions.
- **The judge is lenient.** It credited factually wrong answers and accepted trivial quotes as support. → Decide on the
  deterministic metrics; treat nugget recall and citation support as upper estimates.
- **Citation precision is not interpretable here.** With 2-6 answers per setup there is too little to measure.
- **Retrieval caps everything.** Context coverage ~0.25 limits every setup alike. → Better retrieval (dense or hybrid)
  might be more relevant than adding the context budget.

## Conclusion

The bottleneck in this round is the prompt, not the context budget. Every setup refused most answerable questions,
and the refusals did not depend on whether the context held evidence. Comparing setups only makes sense once the model
answers when it has evidence and refuses when it does not.

**Next: a better prompt.**
- Drop the exact refusal sentence, which the model used as a shortcut. Ask it only to say in one sentence that the
  sources are not enough.
- Test the new prompt on the baseline first, with the same 20-question sample: answered rate and correct
  abstention. It should raise answered questions without losing too much correct abstention.
- If it does, rerun the seven setups with it and choose the final setup from that round.
