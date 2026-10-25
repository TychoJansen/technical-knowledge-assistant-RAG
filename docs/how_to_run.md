# How to run

From a fresh clone to every result in [first_results.md](first_results.md), [second_results.md](second_results.md)
and [third_results.md](third_results.md). The final system is the one in the code: round 3, the baseline setup.

**Rules for every run:**
- **One `rag` command at a time.** Runs compete for the CPU otherwise, and the timings become worthless.
- **Dev until step 7.** The test split is touched once, after the choice is frozen.
- **Interrupted?** Run the same command again: finished questions are skipped. A changed setting is refused; use
  `--restart` (or another `--name`) to start that run over.
- **Every evaluation run keeps its own log** in `logs/` (`answers_<run>_<split>.log`, `abstention_<run>_<split>.log`).
- **Changed the prompt, answer rules or judge?** Move the old runs to `results/archive/<round>/` first (see "Rounds").
  Run names do not contain the prompt, so a new run would otherwise resume or overwrite an old one. The summary
  tables show each run's prompt fingerprint in the "Prompt" column: compare only runs with the same one.

## 1. Install (once)

Needs Python 3.11, [Poetry](https://python-poetry.org) and [Ollama](https://ollama.com). Ollama must be running
(the desktop app, or `ollama serve`) for the CLI and steps 4-7.

```bash
poetry install
ollama pull llama3.2:3b     # assistant
ollama pull llama3          # judge
ollama list                 # both should be listed
```

Ollama runs on another machine or port? Set `OLLAMA_HOST` (default `http://localhost:11434`). No API keys are needed.

## 2. Data and indexes (once)

```bash
poetry run rag prepare                                  # download, dev/test split, statistics
poetry run rag index --tokenizer word_stem
poetry run rag index --tokenizer identifier_split
poetry run pytest                                       # unit tests, no Ollama needed
```

- `rag prepare` needs internet once: it downloads the FreshStack LangChain corpus and queries from Hugging Face.
- The dev/test split is created from a fixed seed, so it is the same every time (60 dev / 143 test questions).
- The indexes take about 15-20 seconds each. `word_stem` is the final retriever; `identifier_split` is only needed for
  setup 3 of the experiments.

## Ask questions (the CLI)

Needs the `word_stem` index (step 2) and a running Ollama with `llama3.2:3b`.

```bash
poetry run rag ask                                   # interactive: type questions; empty line, "exit" or Ctrl+C stops
poetry run rag ask "How do I use a JSON loader?"     # one question, then exit
poetry run rag ask --show-sources                    # also list every retrieved chunk with its BM25 score (* = cited)
```

**Options:** `--top-k` (chunks given to the model, default 4), `--max-chars` (characters per chunk, default 2,000),
`--tokenizer` (default `word_stem`; `identifier_split` needs its own index) and `--model` (Ollama model, default
`llama3.2:3b`).

**Each answer shows:**
- the answer text, or the standard sentence when the knowledge source does not contain enough information;
- the cited sources: each `doc_id` with the GitHub URL of its file;
- a warning when the model's output was invalid, when an answer cites nothing, or when citations to chunks that were
  not retrieved were dropped;
- the model, time and token counts.

The assistant only searches the corpus; it never sees the evaluation questions, answers or labels.

## 3. Retrieval (dev, seconds, no LLM)

```bash
poetry run rag eval retrieval --tokenizer word_stem
poetry run rag eval retrieval --tokenizer identifier_split
poetry run rag eval retrieval --query-mode title                    # shorter queries: tried, not used
poetry run rag eval retrieval --query-mode title_first_paragraph
```

**Outputs:**
- `results/retrieval_summary.md`: all runs, overall and per language. Cov@20 is the primary metric.
- `results/<name>_worst_dev.md`: the 5 questions where retrieval failed most, with what was found and what was
  missed.
- `logs/retrieval_<name>_dev.log`: the run log.

Retrieval does not depend on the prompt, so these results hold for all three rounds.

## Rounds: the three prompts

Steps 4-5 were run three times, once per prompt. Only `SYSTEM_PROMPT` in
[prompt.py](../src/rag_assistant/prompt.py) differs between the rounds; model, retrieval, answer rules (version 4) and
judge (version 5) are the same. The code holds the round-3 prompt.

| Round | Prompt (fingerprint) | Exact text | What was run | Archive |
|---|---|---|---|---|
| 1 | `e988f60a3726` (exact refusal sentence) | [first_results.md](first_results.md), "Prompt" | step 4 (all seven setups), step 5a | `results/archive/round_2026-10-23/` |
| 2 | `c745ef277954` (say so in one sentence) | [second_results.md](second_results.md), "Prompt" | step 4 (all seven setups), step 5a | `results/archive/round_2026-10-24/` |
| 3 | `f2d2db446790` (stricter grounding; final) | [third_results.md](third_results.md), "Prompt" | baseline on the sample, step 5a, step 7 | `results/` |

**To reproduce round 1 or 2:**
1. Move the current runs to `results/archive/<round>/` (keep the `retrieval_*` files, `runs/` and the
   `*_worst_dev.md` files: they do not depend on the prompt).
2. Replace `SYSTEM_PROMPT` in `prompt.py` with that round's text, and check the fingerprint:

   ```bash
   poetry run python -c "from rag_assistant.prompt import prompt_fingerprint; print(prompt_fingerprint())"
   ```

3. Run steps 4 and 5a.
4. Afterwards restore the round-3 prompt (fingerprint `f2d2db446790`).

## 4 + 5a. Run the experiments in one go

Steps 4 and 5a do not depend on any choice, so the script runs them one after another, unattended (~6.5 h):

```bash
poetry run python scripts/run_experiments.py all --dry-run     # see the commands first
poetry run python scripts/run_experiments.py all
```

- **Interrupted, or a command failed?** Start the script again: finished questions and runs are skipped.
- **In parts:** `phase1` runs only step 4 and `phase2a` only step 5a; `phase1 --setups B 1 6` runs only those
  setups.
- **The script runs exactly the commands of steps 4 and 5a below** ([scripts/run_experiments.py](../scripts/run_experiments.py)).
- **Afterwards:** read the results (step 4, "How to choose"), check the judge (step 5b), then decide on step 5c.

## 4. Phase 1: compare the setups (dev sample, without the judge)

Every setup gets two runs on the **same seeded sample of 20 dev questions**:
- an answer run without the judge;
- an abstention run, where each question gets only chunks FreshStack judged not relevant to it, so the right answer
  is "not enough information" ([evaluation.md](evaluation.md), "Abstention").

That is ~15-20 + ~13 minutes per setup, ~4 h for all seven.

| # | Setup | Options | Tests |
|---|---|---|---|
| B | Baseline | (defaults: `word_stem`, top-k 4, 2,000 chars) | reference |
| 1 | Depth | `--top-k 2 --max-chars 4000` | fewer, fuller chunks |
| 2 | Speed | `--top-k 3 --max-chars 1500` | less text |
| 3 | Tokenizer | `--tokenizer identifier_split` | other retrieval |
| 4 | More context | `--top-k 6 --max-chars 2000` | more evidence |
| 5 | Breadth | `--top-k 8 --max-chars 1000` | many short chunks |
| 6 | Less truncation | `--top-k 4 --max-chars 3000` | same chunks, more of each |

**By hand**, for each setup, with its options (`<opts>`; empty for the baseline):

```bash
poetry run rag eval answers --limit 20 --no-judge <opts>
poetry run rag eval abstention --limit 20 <opts>
```

For example, setup 1:

```bash
poetry run rag eval answers --limit 20 --no-judge --top-k 2 --max-chars 4000
poetry run rag eval abstention --limit 20 --top-k 2 --max-chars 4000
```

Then compare each setup with the baseline, on both evaluations:

```bash
poetry run rag eval compare --a bm25_word_stem_llama3.2-3b_k4_c2000_sample20_nojudge --b <setup>_sample20_nojudge
poetry run rag eval compare --abstention --a bm25_word_stem_llama3.2-3b_k4_c2000_sample20 --b <setup>_sample20
```

The run names are `bm25_<tokenizer>_llama3.2-3b_k<k>_c<chars>` plus `_sample20`; answer runs also get `_nojudge`.
→ `results/answer_summary.md`, `results/abstention_summary.md`, and `results/compare_*.json` per comparison.

**Check after setups 4 and 6:** their prompts hold up to 12,000 characters of chunks. The context window is 8,192
tokens (`num_ctx`, the maximum of the judge `llama3`). Search their run logs for `filled the whole context window`.

**How to choose.** These runs show *whether* the model answers, cites and refuses correctly, not *how good* the
answers are. Choose on all of these together, never on one number:

| Metric | Want | Why |
|---|---|---|
| Answered / False abst (answer run) | more answered | answerable questions should be answered |
| Correct abst (abstention run) | higher | the "not enough information" requirement |
| Cit prec (answer run) | higher | citations should point to relevant chunks |
| Ctx Cov (answer run) | higher | more evidence in the context |
| `generation_s_mean` | lower | speed |

- **Answering and refusing pull against each other.** A setup that answers more often usually refuses less, so always
  read Answered and Correct abst together. Rounds 1 and 2 showed both extremes: 20% answered with the exact refusal
  sentence, and 15% correct abstention without it ([experiments.md](experiments.md)).
- **"Answered" is not simply "better".** With context coverage around 0.25, many real contexts hold little of the
  answer, and refusing those is correct, but the answer run counts every refusal as a false abstention. Read Answered
  next to Ctx Cov.
- **Compare only runs with the same prompt fingerprint** (the "Prompt" column of the summary tables).
- **A difference counts only where the 95% interval of `rag eval compare` excludes 0.**
- **Pick the one setup** that is the most promising alternative to the baseline, for step 5c. Write down why. If no
  setup is worth it, skip 5c and keep the baseline.

### Round 3: the baseline on the sample, compared with round 2

Round 3 ran only the baseline (because of the time limit; see [experiments.md](experiments.md)). Before the run,
round 2 was archived, but its two baseline sample runs were copied back into `results/answers/` and
`results/abstention/` so the rounds can be compared per question. The round-3 runs need their own names, because the
default names belong to the round-2 runs:

```bash
poetry run rag eval answers --limit 20 --no-judge --name r3_bm25_word_stem_llama3.2-3b_k4_c2000_sample20_nojudge
poetry run rag eval abstention --limit 20 --name r3_bm25_word_stem_llama3.2-3b_k4_c2000_sample20
poetry run rag eval compare --a bm25_word_stem_llama3.2-3b_k4_c2000_sample20_nojudge --b r3_bm25_word_stem_llama3.2-3b_k4_c2000_sample20_nojudge
poetry run rag eval compare --abstention --a bm25_word_stem_llama3.2-3b_k4_c2000_sample20 --b r3_bm25_word_stem_llama3.2-3b_k4_c2000_sample20
```

## 5. Phase 2: the full dev set (with the judge)

60 dev questions. ~1h30-2h per answer run with the judge, ~40 min per abstention run. The baseline is always in
phase 2 (5a); the setup chosen in step 4 is compared with it (5c).

**5a. The baseline, with the judge** (part of `run_experiments.py all`; alone: `phase2a`).

```bash
poetry run rag eval abstention
poetry run rag eval answers
```

The script then runs an automatic judge check. It reports a problem if:
- the nugget judge failed on more than 10% of the judged answers, or
- citation support is below half of citation precision.

Both are the signs of the output-format defects of judge version 4, fixed in version 5. The check does not replace 5b.

**5b. Judge health check, before judging anything else** ([evaluation.md](evaluation.md), "The judge"). In
`logs/answers_bm25_word_stem_llama3.2-3b_k4_c2000_dev.log` and
`results/answer_bm25_word_stem_llama3.2-3b_k4_c2000_dev.json`, look at:
- **"Nugget judge output incomplete" warnings,** as a share of the answered questions: should be few or none.
- **`citation_support` next to `citation_precision_qrels`:** support far below precision means the judge rejects
  chunks FreshStack marks relevant.
- **`citation_yes_without_quote_rate`:** how often the quote check had to reject a "yes".
- **A few answers and their nugget verdicts, read by eye** (`results/answers/bm25_word_stem_llama3.2-3b_k4_c2000_dev.jsonl`).

If something is wrong: fix the judge, raise `JUDGE_VERSION`, and rerun 5a with `rag eval answers --restart`.

**5c. The setup chosen in step 4, compared with the baseline** (optional; not run in any of the three rounds, the
baseline was kept each time).

```bash
poetry run python scripts/run_experiments.py phase2b --setup 1 --dry-run   # with your setup's label
poetry run python scripts/run_experiments.py phase2b --setup 1             # ~2h20
```

By hand, with the chosen setup's options:

```bash
poetry run rag eval abstention <opts>
poetry run rag eval answers <opts>
poetry run rag eval compare --a bm25_word_stem_llama3.2-3b_k4_c2000 --b <run>
poetry run rag eval compare --abstention --a bm25_word_stem_llama3.2-3b_k4_c2000 --b <run>
```

The run names are `bm25_<tokenizer>_llama3.2-3b_k<k>_c<chars>`, without `_sample20` or `_nojudge` (setup 1:
`bm25_word_stem_llama3.2-3b_k2_c4000`).

Choose between the baseline and this setup on all metrics together, now including Nugget R and Cit support. The judge
is a small local model, not verified and lenient in round 1, so its numbers are upper estimates and an extra signal,
not the deciding one.

**Comparing two rounds on the full dev set.** Both rounds' phase-2a runs have the same name, and `rag eval compare`
only reads `results/answers/` and `results/abstention/`. Copy the archived run in under another name, e.g. for round 2
against round 3:

```bash
cp results/archive/round_2026-10-24/answers/bm25_word_stem_llama3.2-3b_k4_c2000_dev.jsonl results/answers/r2_bm25_word_stem_llama3.2-3b_k4_c2000_dev.jsonl
cp results/archive/round_2026-10-24/abstention/bm25_word_stem_llama3.2-3b_k4_c2000_dev.jsonl results/abstention/r2_bm25_word_stem_llama3.2-3b_k4_c2000_dev.jsonl
poetry run rag eval compare --a r2_bm25_word_stem_llama3.2-3b_k4_c2000 --b bm25_word_stem_llama3.2-3b_k4_c2000
poetry run rag eval compare --abstention --a r2_bm25_word_stem_llama3.2-3b_k4_c2000 --b bm25_word_stem_llama3.2-3b_k4_c2000
```

## 6. Freeze the choice

Write down, **before** looking at test:
- the chosen setup;
- the decision rule;
- the baseline you will compare against.

Change nothing after this point: no prompt, rules, judge or setting.

**Frozen for the final system** ([third_results.md](third_results.md), "Decision"): the baseline (`word_stem`, full
question, top-k 4, 2,000 characters), `llama3.2:3b`, prompt `f2d2db446790`, answer rules version 4, judge `llama3`
version 5.

## 7. Final numbers (test, once)

The frozen configuration is the default, so no options are needed (~5-6 h for 143 questions):

```bash
poetry run rag eval retrieval --split test
poetry run rag eval abstention --split test
poetry run rag eval answers --split test       # with the judge
```

In PowerShell, one after another, stopping if a step fails:

```powershell
poetry run rag eval retrieval --split test; if ($?) { poetry run rag eval abstention --split test }; if ($?) { poetry run rag eval answers --split test }
```

Each command logs a warning that it evaluates on the test split. Report the result as it is; never rerun or tune on
test. → `results/*_test.json`, the `test` rows of the three summary tables, and `logs/*_test.log`.
