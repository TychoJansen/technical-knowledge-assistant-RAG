# Experiments: setups, results and what changed

`scripts/run_experiments.py` runs the comparisons as ordinary `rag` commands, one after another. It only changes CLI
options, never the code:

| Setup | Change | Text budget (characters) | Question it answers |
|---|---|---:|---|
| B (baseline) | top-k 4, 2,000 characters | 8,000 | reference |
| 1 | top-k 2, 4,000 characters | 8,000 | depth: do complete chunks beat more chunks? |
| 2 | top-k 3, 1,500 characters | 4,500 | speed: how much quality is lost per second saved? |
| 3 | `identifier_split` tokenizer | 8,000 | does a slightly better top ranking carry through to the answers? |
| 4 | top-k 6, 2,000 characters | 12,000 | does more evidence help, or only add noise and time? |
| 5 | top-k 8, 1,000 characters | 8,000 | breadth: the opposite of setup 1 at the same budget |
| 6 | top-k 4, 3,000 characters | 12,000 | does cutting chunks less help, with the same chunks? |

Each setup changes one thing, so a difference can be traced to that change. Setups 1 and 5 keep the baseline's budget
and test depth against breadth.

**How the experiment runs:**
1. **Phase 1, screening.** Every setup runs on the same seeded sample of 20 dev questions:
   - an answer run without the judge (deterministic metrics only, 2.5-3.5x faster), and
   - an abstention run.

   Each setup is then compared with the baseline per question (`rag eval compare`), with 95% bootstrap intervals.
2. **Phase 2a.** The baseline runs on all 60 dev questions, with the judge, followed by a check that the judge behaves
   sensibly.
3. **Phase 2b.** The best setup from phase 1 runs on all dev questions and is compared with the baseline. (Not run
   in any round: the baseline was kept each time.)
4. **Test, once.** The chosen setup is frozen and run once on the test split, and nothing is changed based on those
   numbers.

**What keeps the comparisons fair:**
- The same questions in every run.
- Deterministic generation, with the model unloaded before each run.
- Runs one at a time, so timings are not distorted by runs competing for the CPU.
- Every run stores its full configuration, and an interrupted run resumes but refuses changed settings.
- Run names are derived from the settings (e.g. `bm25_word_stem_llama3.2-3b_k2_c4000_sample20_nojudge`), so the runs
  of one round never overwrite each other. The names do not contain the prompt, so each round's runs were archived
  before the next round started.
- With 20 questions, one question is 5 percentage points: trust a difference only if it shows up across several
  metrics and its interval excludes 0.

## Round 1: the prompt with the exact refusal sentence

Details: [first_results.md](first_results.md); the files are in `results/archive/round_2026-10-23/`. In this round
the prompt told the model, when refusing, to *"answer exactly 'The knowledge source does not contain enough
information to answer this question.'"*.

### Phase 1: seven setups (20-question sample, no judge)

| Setup | Ctx Cov | Answered | Correct abstention | Time per question |
|---|---:|---:|---:|---:|
| B baseline | 0.254 | 20% | 70% | 37.2 s |
| 1 depth | 0.183 | **30%** | 70% | 32.5 s |
| 2 speed | 0.196 | 10% | 85% | **22.6 s** |
| 3 `identifier_split` | 0.225 | 15% | 85% | 34.2 s |
| 4 more context | 0.271 | 25% | 85% | 49.6 s |
| 5 breadth | 0.271 | 25% | 70% | 37.7 s |
| 6 less truncation | 0.254 | 15% | 100% | 45.0 s |

- **The model refused 70-90% of the answerable questions in every setup.** All FreshStack questions are answerable
  from the corpus.
- **Only the speed differences were clear.** Setups 1, 2 and 3 were faster than the baseline, and 4 and 6 slower.
  No difference in answered questions or abstention was clear.
- **High "correct abstention" meant little:** setup 6 refused 100% of the insufficient contexts, but also 85% of the
  answerable questions. It simply refused almost everything.
- **Context coverage stayed at ~0.2-0.27 in every setup.** More or longer chunks barely add evidence: the limit is
  what BM25 finds, not how much of it the model sees.

### Phase 2a: the baseline on all 60 dev questions, with the judge

| Metric | Value |
|---|---:|
| Answered | 25% (15 of 60) |
| Correct abstention (insufficient context) | 63% |
| Nugget recall over answered questions (judge) | 0.649 |
| Citation precision (vs FreshStack) | 0.162 |
| Invalid output / uncited answers | 0% / 0% |

**What this showed:**
- **The refusals did not depend on the context.** The model refused the real context (75%) more often than the
  useless one (63%).
- **The exact sentence worked as a shortcut.** The model repeatedly wrote it while its own flag said the information
  was sufficient (the "refused in words" warnings in the run logs): copying, not deciding.
- **The judge's output format was healthy, but its content verdicts were too lenient.** It gave full credit to
  factually wrong answers, and accepted quotes such as `from __future__ import annotations` as "support". Its numbers
  are treated as upper estimates, and decisions are made on the deterministic metrics.
- **No setup could be chosen:** the over-refusal swamped all setup differences.

## What changed after round 1

**1. The prompt no longer dictates the refusal sentence.** The instruction became "Then say so in one sentence and
cite nothing." The user still always sees the standard sentence, because the answer rules give it to every refusal.
This was the smallest change that tests the hypothesis: one sentence, with model, retrieval, rules and judge unchanged.

**2. Quick check on the baseline (same 20 questions):**

| | Exact sentence | New prompt |
|---|---:|---:|
| Answered | 4/20 (20%) | **17/20 (85%)** |
| Correct abstention (insufficient context) | 14/20 (70%) | **3/20 (15%)** |

- **The hypothesis was confirmed:** without the ready-made sentence, the model answers 85% instead of 20%.
- **But it now answers almost everything,** including 16 of the 20 contexts that cannot answer the question, often
  from its own knowledge with citations that do not back the answer up.
- **The model does not judge sufficiency; it follows a default.** With one prompt it refuses almost everything, with
  the other it answers almost everything.

**3. Decision: keep the new prompt and report the weak abstention as a limitation.** Answering 85% of the answerable
questions serves the user better than refusing 80% of them. The alternatives:
- **A middle wording:** likely to only shift the default again.
- **A BM25 relevance gate:** already shown not to work; the score reflects question length, not relevance.
- **"Evidence first":** the model quotes its evidence, and code refuses when no quote is found. This is the most
  promising, but it needs a new prompt, answer format and test cycle, so it remains future work.

**4. Rerun of the round with the new prompt.** All round-1 runs are archived and none can be reused, because their
prompt fingerprint differs.

## Round 2: the prompt without the exact sentence

Details: [second_results.md](second_results.md); the files are in
`results/archive/round_2026-10-24/`.

### Phase 1: seven setups (20-question sample, no judge)

| Setup | Ctx Cov | Answered | Correct abstention | Cit prec | Time per question |
|---|---:|---:|---:|---:|---:|
| B baseline | 0.254 | 85% | 15% | 0.14 | 43.8 s |
| 1 depth | 0.183 | 85% | **25%** | **0.23** | 38.8 s |
| 2 speed | 0.196 | 85% | 15% | 0.19 | **29.8 s** |
| 3 `identifier_split` | 0.225 | 80% | 20% | 0.13 | 44.3 s |
| 4 more context | 0.271 | 75% | 15% | 0.20 | 67.1 s |
| 5 breadth | 0.271 | **95%** | 5% | 0.14 | 54.6 s |
| 6 less truncation | 0.254 | 85% | 5% | 0.15 | 53.7 s |

- **The over-refusal of round 1 is gone:** every setup answered 75-95% of the questions.
- **Abstention collapsed instead:** every setup answered 75-95% of the insufficient contexts, and 93-100% of those
  answers cite a non-relevant chunk.
- **Only speed differences were clear:** setup 2 was 14 s per question faster; setups 4, 5 and 6 were 10-23 s slower.
  No difference in answered questions, abstention or citation precision was clear.
- **Choice:** setup 2 was the best alternative (faster, no measurable loss), but the baseline was kept, so phase 2b
  was not run.

### Phase 2a: the baseline on all 60 dev questions, with the judge

| Metric | Value |
|---|---:|
| Answered | 77% |
| Correct abstention (insufficient context) | 17% |
| Nugget recall, all questions / over answered questions (judge) | 0.41 / 0.54 |
| Citation precision / citation support (judge) | 0.21 / 0.20 |
| Answers without a valid citation | 2% |

- **The sample result held:** the model answers almost everything, including 82% of the insufficient contexts.
- **The judge was healthy:** no parse failures, and citation support close to citation precision.

## What changed after round 2

**1. The prompt contradicted itself.** It said "use ONLY the sources", but to refuse "ONLY if none of the sources is
about the question's topic". The insufficient contexts are on-topic chunks that do not contain the answer, so
answering them followed the prompt. "Never answer from outside knowledge" was also part of the refusal rule, not a
general rule, and citations were not tied to statements.

**2. The round-2 decision was revisited.** Reporting the weak abstention as a limitation was the plan, but this
contradiction was a concrete, testable cause. The smallest change that tests it: rewrite the prompt only, with model,
retrieval, answer rules, schema and judge unchanged.

**3. What would count as better** (against the round-2 baseline on the same 20 questions): correct abstention
clearly higher, while the model still answers well above round 1's 20%.

**4. Only the baseline was rerun, because of the time limit.** Rerunning all seven setups takes about 4 hours, plus
about 3 hours for a full-dev confirmation of a winner. In round 2 the setups differed clearly only in speed, and a
rerun would use the same 20 questions once more.

**Worth testing with more time: more chunks.** With the round-3 prompt the model refuses when the sources do not
contain the answer, so how much evidence reaches it should now matter more than in round 2. Two effects pull in
opposite directions:
- **More evidence could mean fewer wrong refusals.** BM25 finds about half of the key facts in its top 20
  (Cov@20 0.53), but only a quarter reach the model with 4 chunks (context coverage 0.26). More chunks could turn
  refusals for lack of evidence into answers. The gain from 6 or 8 chunks is likely small, though: in round 2 they
  raised context coverage only to 0.27, because of the key facts BM25 finds in its top 20, about half rank between 9
  and 20.
  Reaching those needs more and shorter chunks, e.g. 12-16 chunks of 500-700 characters, to stay within the context
  window. Key facts that BM25 does not rank in the top 20 at all (about half) need better retrieval, not more chunks.
- **More on-topic noise could mean more guessing.** Every extra chunk is another loosely related source that can look
  like an answer. In round 2, the setups with the most chunks (5: 8 chunks) and the longest
  chunks (6: 3,000 characters) abstained worst (5%).

Which effect wins is untested. The cost side is known: more text makes every answer slower (setups 4-6: +10 to +23 s
per question in round 2), and 6 chunks of 2,000 characters already come close to the 8,192-token context window.
Since the test split has been used, such a run would be a post-hoc experiment on dev, not a
change to the final configuration.

## Round 3: the stricter grounding prompt

Details: [third_results.md](third_results.md). The prompt asks to answer only with information
from the sources, to refuse when they do not contain the answer *"even if the sources are about the same topic"*,
and to support every statement with a cited source.

### The sample: the baseline on the same 20 dev questions

| Metric | Round 2 | Round 3 | Round 3 − round 2 |
|---|---:|---:|---|
| Correct abstention | 15% | **60%** | **+0.45**, clear |
| Answered | **85%** | 55% | **−0.30**, clear |

Correct abstention rose clearly while the model still answers more than half of the questions, so the new prompt
went to phase 2a. The rule for that run, set beforehand: keep the new prompt if correct abstention stays clearly
higher without a clear loss of nugget recall. (First formulated for the questions whose context holds evidence; the
repository's evaluation does not compute that split, so nugget recall over all questions is used, the stricter test.)

### Phase 2a: the baseline on all 60 dev questions, with the judge

| Metric | Round 2 | Round 3 | Round 3 − round 2 |
|---|---:|---:|---|
| **Correct abstention** | 17% | **50%** | **+0.33**, clear |
| Answered | **77%** | 52% | **−0.25**, clear |
| **Nugget recall, all questions** | **0.41** | 0.32 | −0.10, not clear (borderline) |
| Nugget recall over answered questions | 0.54 | **0.61** | |
| Citation precision / support | 0.21 / 0.20 | 0.15 / 0.23 | not clear |
| Answers without a valid citation | **2%** | 16% | |

- **The decision rule was met, narrowly:** abstention clearly higher, and the drop in nugget recall is not clear
  (95% interval [−0.20, +0.00]).
- **The model refuses much more often when it should:** correct abstention tripled.
- **The price:** a quarter fewer answers, some of which the judge would have credited, and more answers that forget
  to cite.
- **Decision: the round-3 prompt is the final prompt.**

### Test: the frozen configuration on 143 held-out questions, run once

| Metric | Dev (60) | Test (143) |
|---|---:|---:|
| Correct abstention | 50% | **46%** |
| Answered | 52% | **58%** |
| Nugget recall over answered questions (judge) | 0.61 | **0.48** |
| Citation precision / support | 0.15 / 0.23 | **0.28 / 0.37** |
| Answers without a valid citation | 16% | 18% |
| Retrieval Cov@20 | 0.53 | 0.53 |

- **Abstention generalizes** (46% against 50% on dev).
- **Answer quality is lower than on dev,** partly the expected optimism of the set the prompt was chosen on.
- **Guessing is reduced, not solved:** 54% of the insufficient contexts are still answered.
- **Retrieval stays the cap:** about half of the key facts are not in BM25's top 20 (Cov@20 0.53).

## The final configuration

- **Assistant:** `llama3.2:3b`, temperature 0, seed 42, at most 800 tokens, `num_ctx` 8192.
- **Retrieval:** BM25 `word_stem` on the full question; top-k 4, at most 2,000 characters per chunk.
- **Prompt:** round 3 (`f2d2db446790`), with answer rules version 4.
- **Judge:** `llama3`, judge version 5.

**Not tested:** the context budget with the round-3 prompt (only the baseline was run), the flag before the answer in
the output schema, and "evidence first".
