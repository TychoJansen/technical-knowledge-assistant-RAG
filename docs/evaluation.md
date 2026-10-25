# Evaluation: choices and metrics

## What is evaluated

| Command | Question it answers | Uses an LLM judge? |
|---|---|---|
| `rag eval retrieval` | Does BM25 find the evidence? (top 100 per question) | No |
| `rag eval answers` | Does the assistant answer, from the right evidence, with correct citations? | Optional (`--no-judge`) |
| `rag eval abstention` | Does it say the information is insufficient when the context cannot answer? | No |
| `rag eval compare` | Is the difference between two runs real or noise? | No |

All use the dev split by default. Test is only for final numbers, and evaluating on test logs a warning.

### Retrieval metrics

| Metric | Why |
|---|---|
| **Coverage@20** (primary) | The share of a question's nuggets with at least one supporting chunk in the top 20. A RAG answer needs evidence for each key fact, not every relevant chunk. It is also FreshStack's own metric |
| Recall@10/50/100 | How much of the relevant set is found. R@50/100 is the candidate pool a reranker could work with |
| nDCG@10, MRR@10 | Ranking quality at the top, where the model's context comes from |
| **Unjudged@10** | Not a quality metric: the share of the top 10 that was never judged. It shows how far recall may be underestimated |
| Latency (mean, p95) | Practical cost per question |

- **Macro-averaged:** every question counts equally, regardless of how many relevant chunks it has.
- **Every run stores its top 100 per question,** so later work such as fusion or a reranker can start from the stored
  runs, and the five worst questions are written out for error analysis.

### Answer metrics

| Metric | Kind | Why |
|---|---|---|
| **Ctx Cov / Ctx R** | Deterministic | Coverage and recall of the k chunks the model actually got: the ceiling for a grounded answer |
| **Answered / False abst / Invalid** | Deterministic | Every FreshStack question is answerable, so a refusal is a false abstention; invalid output shows broken generation |
| **Cit prec** | Deterministic | The share of citations FreshStack marks relevant: are citations pointing at real evidence? |
| Uncited answers, dropped / repaired citations | Deterministic | Is every answer grounded in a source it was given? |
| **Nugget recall** (all questions, and answered only) | Judge | Does the answer contain the key facts? Checking facts fits better than matching the wording of a reference answer, which may use an outdated API |
| **Cit support** | Judge + check | Does the cited chunk back up this answer? It counts only if the judge says yes and its quote is verified to occur in the chunk |
| Generation and judge time, tokens | Deterministic | Cost next to quality |

**Deterministic metrics come first.** They are exact and free. The judge metrics add answer quality but depend on a
small model, so they are a second signal.

### The judge

- **`llama3` (8B), larger than and different from the assistant (`llama3.2:3b`),** so the assistant does not grade
  its own answers.
- **Nuggets: one call per question,** with one required true/false field per fact, so the judge cannot skip a fact.
- **Citations: one call per cited chunk.** The chunk is cut exactly as in the assistant's prompt, so the judge sees
  what the assistant saw.
- **Quote verification.** A quote counts only if at least 80% of it occurs as one block in the chunk (after
  normalising case and whitespace), and it is at least 15 characters long. This turns the judge's opinion into
  checkable evidence and rejects paraphrases.
- **Degenerate-answer floor.** An answer under 20 characters, or one that only repeats the title, gets no credit,
  without a judge call.
- **Deterministic** (temperature 0, fixed seed), and the judge version is stored with every run.
- **The judge may see nuggets.** It grades the assistant; nothing it sees reaches the assistant.

### Abstention

FreshStack has only answerable questions, so insufficient information is built from its own labels:
- **The context:** each question gets only chunks the assessors judged **not relevant** to it. Chunks that are
  relevant to any nugget, or that have identical text to a relevant chunk, are excluded. BM25 ranks the remaining
  chunks, so the context looks as much like evidence as possible.
- **The metrics:**
  - **Correct abst:** the model says the information is insufficient.
  - **False answer:** it answers anyway.
  - **Cited false ans:** it answers anyway and cites a non-relevant chunk as its source.
- **Why this test:** the questions are real and on-topic, it follows the dev/test split, and it needs no judge.
- **How to read it:** always next to Answered, because a model that refuses everything scores 100%.

### Comparing runs and keeping them fair

- **Paired comparison per question,** with a 95% bootstrap interval (10,000 resamples, seed 42). A difference counts
  only if the interval excludes 0.
- **A seeded random sample** for `--limit`, so every setup answers the same questions.
- **Reproducible runs:**
  - Every run stores its full configuration: models with digests, prompt fingerprint, answer-rules version, judge
    version, sample and settings.
  - A resumed run with a different configuration is refused.
  - The model is unloaded before each run, which clears Ollama's prompt cache.
- **Resumable runs:** records are written per question, so an interrupted run continues where it stopped.

## Limitations and alternatives

These follow from the metric and evaluation design itself.

- **The labels are incomplete.** Only ~68 chunks per question are judged, and everything else counts as not
  relevant. Recall, coverage and citation precision are lower bounds, and a retriever unlike FreshStack's pool is
  penalised. → Judge a sample of unjudged top hits with an LLM, and always report Unj@10.
- **The nuggets are GPT-4o-generated.** They may miss or over-specify facts. → Check a sample by hand.
- **Ctx Cov ignores truncation.** A chunk counts as evidence even when the supporting part was cut off before the
  model saw it. → Compute coverage on the text actually shown.
- **Nugget recall does not penalise wrong statements.** The judge only checks whether a fact is present, so an answer
  that also contradicts it, or adds invented APIs, can still score high. → A correctness or contradiction check, or a
  claim-level precision ("which claims are supported by the sources?").
- **The judge is a small, unverified model.** Its verdicts vary with wording and model. → A stronger judge, or
  agreement with a small hand-labelled set.
- **The quote check proves a quote exists, not that it supports the claim.** An import line passes. → Require the
  quote to share content with the claim, and reject trivial quotes such as imports and bare parameter values.
- **Citation precision measures relevance to the question, not support for the answer.** A relevant chunk can be
  cited for a claim it does not make. → Read it next to Cit support.
- **False abstention mixes retrieval and model failures.** A refusal with no evidence in the context is honest. →
  Report false abstentions split by Ctx Cov = 0 versus > 0.
- **Abstention is tested with one kind of insufficient context only:** on-topic, judged-not-relevant chunks, all
  insufficient. It is strict, because a loosely useful chunk still counts as "not relevant", and it does not test
  mixed contexts. Correct abstention alone says little. → Always read it next to Answered. A per-question pattern
  (answers the real context, refuses the insufficient one) would be a stronger measure, but `rag eval` does not compute
  it yet.
- **The samples are small.** With 20 questions, one question is 5 points. The bootstrap shows the noise but cannot
  remove it, and a sample's absolute numbers need not match the full set. → Compare on the sample; report absolute
  numbers from the full dev set and test.
- **Citation metrics are micro-averaged,** so answers with many citations weigh more. → Add a per-answer
  (macro) variant.
