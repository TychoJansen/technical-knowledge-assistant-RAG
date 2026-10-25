# technical-knowledge-assistant-RAG

A retrieval-augmented assistant for technical LangChain questions. It searches only the FreshStack LangChain corpus,
answers with a local LLM, cites the supporting chunks by `doc_id`, and says so when the corpus does not contain
enough information. Everything runs locally and for free: BM25 retrieval, `llama3.2:3b` via Ollama, no API keys.

## 1. Repository structure

```
src/rag_assistant/
  cli.py            the `rag` command: prepare, index, ask, eval
  prepare.py        download the data, fixed dev/test split, statistics
  data.py           loaders for the local data files (corpus = knowledge; queries = evaluation only)
  tokenizers.py     BM25 tokenizers (word_stem, identifier_split)
  retrieval.py      BM25 retriever and its on-disk index
  query.py          optional shorter search queries (tried, not used)
  prompt.py         system prompt, answer schema, source formatting
  llm.py            LLM interface and the Ollama backend
  assistant.py      retrieve -> prompt -> LLM -> checked citations and refusals
  config.py         paths and shared defaults (the final configuration)
  logger.py
  evaluation/       retrieval, answer and abstention evaluation, LLM judge, metrics, run comparison
scripts/run_experiments.py   runs the setup comparisons one after another
tests/              unit tests (no Ollama needed)
docs/               design choices, experiments and results
data/  indexes/  results/  logs/   generated (data/splits.json and data/stats.json are committed)
```

## 2. Docs

| Doc | Content |
|---|---|
| [data_preparation.md](docs/data_preparation.md) | How the data is prepared and split; what the corpus and queries look like and what that means for the design |
| [retrieval.md](docs/retrieval.md) | BM25 and the tokenizers: choices, retrieval results, limitations and alternatives |
| [generation.md](docs/generation.md) | The assistant: model choice (models tried), prompt, answer rules, context budget |
| [evaluation.md](docs/evaluation.md) | What is measured and why: retrieval, answer, citation and abstention metrics; the judge |
| [experiments.md](docs/experiments.md) | The setups and the three prompt rounds in one overview, and what changed after each |
| [first_results.md](docs/first_results.md) | Round 1: seven setups with the "exact refusal sentence" prompt |
| [second_results.md](docs/second_results.md) | Round 2: seven setups with the "say so in one sentence" prompt |
| [third_results.md](docs/third_results.md) | Round 3: the stricter grounding prompt, the full dev set, and the final test results with examples |
| [how_to_run.md](docs/how_to_run.md) | Every command, from a fresh clone to the results of all three rounds |

## 3. How to run the final model

Needs Python 3.11, [Poetry](https://python-poetry.org) and a running [Ollama](https://ollama.com). No API keys; set
`OLLAMA_HOST` only if Ollama is not on `http://localhost:11434`.

```bash
# 1. Install
poetry install
ollama pull llama3.2:3b      # assistant
ollama pull llama3           # judge

# 2. Data and index (internet needed once)
poetry run rag prepare                       # corpus, questions, fixed dev/test split
poetry run rag index --tokenizer word_stem   # BM25 index

# 3. Ask questions
poetry run rag ask                                   # interactive; empty line or "exit" stops
poetry run rag ask "How do I use a JSON loader?"     # one question
poetry run rag ask --show-sources                    # also list every retrieved chunk

# 4. The final results on the test split (the final configuration is the default)
poetry run rag eval retrieval --split test
poetry run rag eval abstention --split test
poetry run rag eval answers --split test     # with the judge
```

The results are in `results/`: `retrieval_summary.md`, `answer_summary.md` and `abstention_summary.md` (one row per
run), and the per-run reports `*_test.json`. An interrupted run continues where it stopped when you run the same
command again. Unit tests: `poetry run pytest`. To reproduce the experiments and the earlier rounds, see
[docs/how_to_run.md](docs/how_to_run.md).

## 4. Approach

1. **Data.** `rag prepare` downloads the FreshStack LangChain corpus (49,514 chunks from 10 repositories) and the 203
   Stack Overflow questions with their nuggets (key facts) and relevance judgments. The corpus is the only knowledge
   source; the questions, answers and labels are used only for evaluation. A fixed, seeded split, stratified by
   language, gives **60 dev questions** for all choices and **143 test questions**, held out until the end.
2. **Retrieval.** BM25 (`bm25s`, default parameters) over FreshStack's chunks as they are, searched with the full
   question. Lexical search suits questions full of identifiers and error messages, is fast (~3 ms per query) and
   needs no GPU. Shorter queries (title, title + first paragraph) were tried and were worse.
3. **Tokenizer.** `word_stem` (lowercase, stopwords removed, stemmed) against `identifier_split` (also splits
   `camelCase` and `snake_case`). They were equal within the noise (Cov@20 0.53 against 0.50 on dev), so the simpler
   and faster `word_stem` was kept.
4. **Model selection.** Three local models were tried as the assistant on the same questions. `llama3.2:3b` was the
   only one with valid JSON, exact citations and a correct refusal, and the fastest. `qwen2.5:3b` answered an
   unanswerable question from its own knowledge; `llama3` (8B) cited a chunk it was never given and was 2-3x slower.
   `llama3` became the judge instead: larger than and different from the assistant.
5. **Answering.** The top 4 chunks (each cut to 2,000 characters) and the question go to the model, which must return
   JSON: answer, cited `doc_id`s and an `insufficient_information` flag. Code then drops citations to chunks the model
   was not given, repairs obvious copying errors, and gives every refusal the same sentence.
6. **Experiments.** Seven setups (number of chunks, chunk length, tokenizer) were compared on a seeded sample of 20
   dev questions, without the judge; the baseline then ran on all 60 dev questions with the judge. This was done for
   three prompts, because the prompt, not the setup, turned out to decide the behaviour:

   | Round | Prompt change | Answered (dev, 60) | Correct abstention (dev, 60) |
   |---|---|---:|---:|
   | 1 | Refuse with an exact sentence | 25% | 63% |
   | 2 | "Say so in one sentence" | 77% | 17% |
   | 3 | Answer only from the sources; refuse when they do not contain the answer, "even if they are about the same topic" | 52% | 50% |

   Round 1 refused almost everything, round 2 answered almost everything. Round 3 was the first prompt that made the
   model refuse much more often when it should, without a clear loss of nugget recall, so it became the final prompt.
   In rounds 1 and 2 the setups differed clearly only in speed, so the baseline setup was kept.
7. **Dev and test.** Every choice was made on dev. The final configuration was frozen and run **once** on the test
   split: `word_stem`, full question, top 4 chunks of 2,000 characters, `llama3.2:3b` (temperature 0, seed 42), the
   round-3 prompt (`f2d2db446790`), judge `llama3`.

## 5. Results of the final model

The frozen configuration on the 143 held-out test questions, with dev for comparison. The assistant runs every
question twice: with its own retrieved context (answer run) and with only chunks FreshStack judged *not* relevant to
the question, where the correct outcome is a refusal (abstention run).

**Retrieval** (BM25, no LLM):

| Metric | Dev (60) | Test (143) |
|---|---:|---:|
| **Cov@20**: share of key facts with supporting evidence in the top 20 | 0.53 | **0.53** |
| R@10 / R@100: share of relevant chunks found | 0.17 / 0.41 | 0.18 / 0.37 |
| nDCG@10 / MRR@10 | 0.19 / 0.36 | 0.22 / 0.40 |
| Unjudged share of the top 10 (recall is a lower bound) | 0.45 | 0.43 |

**Answers, citations and abstention:**

| Metric | Dev (60) | Test (143) |
|---|---:|---:|
| **Answered** (all questions are answerable from the corpus) | 52% | **58%** |
| **Correct abstention** (refuses when the context cannot answer) | 50% | **46%** |
| Context coverage: key facts in the 4 chunks the model sees | 0.26 | 0.32 |
| **Nugget recall over answered questions** (judge) | 0.61 | **0.48** |
| Nugget recall over all questions (judge) | 0.32 | 0.28 |
| **Citation precision**: cited chunks FreshStack judged relevant (lower bound) | 0.15 | **0.28** |
| **Citation support**: judge confirms the chunk backs the answer, with a verified quote | 0.23 | **0.37** |
| Answers without a valid citation | 16% | 18% |
| False answers on insufficient context that cite a non-relevant chunk | 86% | 84% |
| Invalid output (answer run / abstention run) | 0% / 2% | 0% / 0.7% |
| Time per question: generation + judge | 39 s + 52 s | 38 s + 56 s |

**Interpretation:**
- **Abstention generalizes:** 46% on test against 50% on dev, about three times the round-2 prompt (17% on dev). But
  the model still answers 54% of the contexts that cannot answer the question, almost always citing an irrelevant
  chunk. Guessing is reduced, not solved.
- **Answer quality is lower than on dev** (nugget recall 0.48 against 0.61 over answered questions): numbers measured
  on the set the prompt was chosen on are optimistic.
- **Citations are better on test** (precision 0.28, support 0.37), in line with more evidence in the context (0.32
  against 0.26). Both are rough: unjudged chunks count as not relevant, and the judge is lenient.
- **Retrieval is the cap:** only about a third of the key facts reach the model, and about half are not even in
  BM25's top 20.
- **No technical failures:** no invalid answers, no prompt that filled the context window, no judge parse failures.

**Examples from the test run:**
- **Good: answered from evidence, with a supporting citation.** *"Langchain Pandas dataframe agent answering questions
  through google search"*: the answer combines the tools of the csv-agent template and cites that template. The
  judge credits all 5 key facts with a verified quote, and FreshStack also judges the chunk relevant.
- **Good: refuses the insufficient context, answers the real one.** *"How can I limit sourceDocs in langchain +
  pinecone response in JS?"*: the standard refusal with non-relevant chunks; with the real context the one-word answer
  `returnSourceDocuments`, to which the lenient judge still credits 2 of 3 key facts.
- **Poor: a guess on an insufficient context.** *"What does `pip install unstructured[local-inference]` do?"*: given
  only non-relevant chunks, it claims the feature means "the model is trained on a specific dataset" and cites a
  documentation-setup page that says nothing of the kind.
- **Poor: refuses although the evidence is there.** *"Can't set custom OpenAI model with langchain on nodejs"*: two
  thirds of the key facts are in the context, and the model refuses.
- **Poor: a plausible answer without evidence.** *"Databricks Dolly LLM: empty result when using LangChain with
  context"*: no relevant chunk in the context; the model invents an explanation, cites an unrelated notebook, and the
  judge credits none of the key facts.

Full tables and per-question comparisons: [docs/third_results.md](docs/third_results.md).

## 6. Trade-offs

**Round 3, abstention against answering** (dev, 60 questions, against the round-2 prompt):

| | Round 2 | Round 3 (final) |
|---|---:|---:|
| Correct abstention | 17% | **50%** (clear) |
| Answered | **77%** | 52% (clear) |
| Nugget recall, all questions | **0.41** | 0.32 (not clear, borderline) |
| Nugget recall over answered questions | 0.54 | **0.61** |
| Answers without a valid citation | **2%** | 16% |
| Judge time per question | 70 s | **52 s** |

Being honest about missing information costs answers: some of the refused questions would have been answered
correctly from the model's own knowledge. Round 3 chooses grounding over coverage at the price of more uncited answers.

**The context budget** (round 2, 20 dev questions; only the speed differences were clear):

| Setup | Text per question | Time per question | Trade-off |
|---|---:|---:|---|
| Baseline: 4 chunks × 2,000 characters (final) | 8,000 | 43.8 s | reference |
| 2: 3 × 1,500 | 4,500 | **29.8 s** (−14 s, clear) | ~30% faster, no measurable loss in answering or abstention, but slightly less evidence |
| 1: 2 × 4,000 | 8,000 | 38.8 s | most precise citations (0.23), but the least evidence |
| 4: 6 × 2,000 | 12,000 | 67.1 s (+23 s, clear) | barely more evidence (0.27 against 0.25) |
| 5: 8 × 1,000 | 8,000 | 54.6 s (+11 s, clear) | barely more evidence; worst abstention (5%) |
| 6: 4 × 3,000 | 12,000 | 53.7 s (+10 s, clear) | same chunks, more of each; worst abstention (5%) |

More text makes every answer slower without adding much evidence, because the limit is what BM25 finds. Setup 2 is
the best choice when speed matters; the baseline was kept because it keeps the most evidence at moderate speed.

**Speed and cost:**
- **Free, local and slow.** No API costs or keys, but ~38 s per answer on a laptop CPU, and ~5-6 h for the full test
  evaluation with the judge (which takes more time than generating). A hosted model would answer in seconds and
  probably cite and refuse better, at a cost per call and with rate limits.
- **3B against 8B.** `llama3` (8B) was 2-3x slower as the assistant and did not cite more accurately; a 3B model
  keeps interactive use possible.
- **The judge.** `qwen2.5:3b` judged 2.5x faster than `llama3`, but barely agreed with it (kappa 0.18), so the slower
  judge was kept.
- **Retrieval is nearly free:** ~3 ms per query and ~20 s to build the index; `identifier_split` costs a bit more
  for no gain.

## 7. Limitations

**The final model:**
- **It still guesses.** It answers about half of the contexts that cannot answer the question. The decision to refuse
  is made by a 3B model in one pass, and the flag is generated *after* the answer.
- **Citations are checked to exist, not to support.** Code guarantees that a cited `doc_id` was given to the model,
  not that it backs the answer; 18% of the answers cite nothing at all.
- **It knows LangChain from training.** "Only use the sources" cannot be enforced: the model cannot tell its own
  knowledge from the sources.
- **The judge is a small, unverified model.** It credits wrong or one-word answers, and nugget recall does not penalise
  invented statements. Nugget recall and citation support are upper estimates.
- **Chosen on a small dev set.** The prompt was chosen on 20 questions (used three times) and confirmed on 60; the
  context budget was only compared with the round-2 prompt.

**The retriever:**
- **Half of the key facts are not in the top 20** (Cov@20 0.53), and only a quarter to a third reach the model with
  4 chunks. More or longer chunks barely help (context coverage 0.18-0.27 across setups).
- **Vocabulary mismatch:** users describe symptoms ("stream a response"), the docs use other terms (callbacks).
- **Long, noisy questions:** pasted logs and package lists match unrelated files such as `requirements.txt`.
- **Noisy corpus:** notebooks are indexed as raw JSON, and 13% of the chunks are exact duplicates.
- **Incomplete labels:** ~43% of the top 10 was never judged, so recall, coverage and citation precision are lower
  bounds.

## 8. Lessons learned

**The prompt decided the behaviour, more than any setup.**
- **One sentence flips a small model between two extremes.** Telling it to refuse with an exact sentence (round 1)
  made it answer 25% of the dev questions; asking it to "say so in one sentence" (round 2) made it answer 77%, but
  correct abstention fell from 63% to 17%. Neither prompt made it refuse selectively; each only set a default.
- **Do not dictate output text.** In round 1 the model copied the ready-made refusal sentence as a shortcut, even
  while its own flag said the information was sufficient. A fixed sentence belongs in code, not in the prompt.
- **A contradiction in the prompt is resolved in favour of answering.** The round-2 prompt said "use ONLY the sources",
  but to refuse "ONLY if none of the sources is about the question's topic". The insufficient contexts are on-topic,
  so answering them followed the prompt. Making the threshold explicit in round 3 ("refuse when the sources do not
  contain the answer, even if they are about the same topic") tripled correct abstention (17% to 50% on dev).
- **Grounding has a price.** The stricter prompt answers a quarter fewer questions, some of which the judge would have
  credited, and 16-18% of its answers forget to cite. Instructions alone do not guarantee citations; only checks in
  code do.

**The number and length of chunks mattered much less than expected.**
- **More text hardly adds evidence.** Across all seven setups and both rounds, context coverage stayed at 0.18-0.27:
  4, 6 or 8 chunks, cut at 1,000 to 4,000 characters, all contain about a quarter of the key facts. The limit is what
  BM25 ranks high, not how much of each chunk the model reads.
- **More text clearly costs time; less text clearly saves it.** Setups with 12,000 characters were 10-23 s per
  question slower; 3 chunks of 1,500 characters were 14 s faster with no measurable loss.
- **Each direction has its own risk.** Fewer, longer chunks gave the most precise citations but the least evidence;
  more chunks, or longer ones, gave the model more on-topic noise: 8 chunks of 1,000 characters and 4 chunks of 3,000
  characters abstained worst (5%).
- **Fix the behaviour before tuning the budget.** In round 1 the over-refusal hid every setup difference, so the setups
  could only be compared once the prompt worked.

**Retrieval is the real bottleneck.**
- BM25 finds supporting evidence for only about half of the key facts in its top 20 (Cov@20 0.53 on dev and test), and
  only a quarter to a third reach the model. The other tokenizer and shorter queries did not help; the failures are
  vocabulary mismatch and long, noisy questions.

**Evaluation.**
- **Abstention needs its own test.** Without contexts that cannot answer the question, round 2 would have looked like
  the best prompt (77% answered). Correct abstention and Answered must always be read together.
- **Decide the rule before the run, and use test once.** A 20-question sample only shows large differences (one
  question is 5 points); the prompt was chosen on dev, and test answer quality turned out lower than dev's.
- **A small judge is lenient.** It credited one-word and wrong answers, so the deterministic metrics decided.

## 9. Next steps

In order of expected impact: retrieval first, because the model cannot ground an answer in evidence it never sees.
Each step changes one thing and is compared on dev against the final configuration with `rag eval compare`. The test
split has been used once, so a new final number needs a fresh held-out set.

1. **Better retrieval** (goal: more key facts in the top 4-10).
   - **Hybrid search:** BM25 plus dense embeddings (e.g. `nomic-embed-text` or `bge-m3` via Ollama), merged with
     Reciprocal Rank Fusion. Embeddings fix the vocabulary mismatch; BM25 keeps exact identifiers. It plugs in as a new
     `Retriever` in `retrieval.py`. *Alternative:* dense only, simpler but weaker on identifiers.
   - **A reranker** on the top 50-100, e.g. the cross-encoder `bge-reranker-v2-m3`. *Alternative:* let the LLM score
     the candidates, which needs no new dependency but is slow.
   - **Cleaner input:** index notebook cell text instead of raw JSON, collapse duplicates, add the file path to the
     indexed text, and strip pasted logs from long questions (or let the LLM rewrite the query).
2. **The context budget with the round-3 prompt.** Try more and shorter chunks (e.g. `--top-k 12 --max-chars 700`) to
   reach the evidence at ranks 9-20, and show the best-matching part of each chunk instead of its first 2,000
   characters. Judge on correct abstention and Answered together, since more text can also mean more guessing.
3. **Make refusing depend on evidence.**
   - **Flag first:** put `insufficient_information` before the answer in the schema, so the model decides before it
     writes.
   - **"Evidence first":** the model quotes the lines it relies on, and code refuses when no quote is found in a source
     (the judge's `quote_in_source` can be reused).
   - **Short source ids** (`[1]`-`[4]`) to avoid copying errors, and **at least one valid citation** per answer.
   - *Alternatives:* a second LLM call that checks the answer against the sources, or a gate on the reranker score.
4. **A stronger evaluation:** compute the evidence split and the per-question abstention pattern in `rag eval`, check
   the judge against 30-50 hand-labelled answers, and add a check for statements the sources do not support.
5. **A stronger model:** a 7-8B local model on a GPU (free, needs hardware), or a hosted model such as Claude or GPT
   (seconds per question and better at citing and refusing, but costs per call and needs an API key).

## Dataset

The LangChain subset of [FreshStack](https://arxiv.org/abs/2504.13128) (October 2024 snapshot):
[`freshstack/corpus-oct-2024`](https://huggingface.co/datasets/freshstack/corpus-oct-2024) and
[`freshstack/queries-oct-2024`](https://huggingface.co/datasets/freshstack/queries-oct-2024).

```bibtex
@misc{thakur2025freshstack,
      title={FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents},
      author={Nandan Thakur and Jimmy Lin and Sam Havens and Michael Carbin and Omar Khattab and Andrew Drozdov},
      year={2025},
      eprint={2504.13128},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.13128},
}
```
