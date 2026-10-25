# Generation: model and assistant choices

## How an answer is made

1. **Retrieve:** BM25 returns the top 4 chunks for the question.
2. **Prompt:** each chunk is wrapped as `<source doc_id="...">` and cut to 2,000 characters; the question follows the
   sources.
3. **Generate:** `llama3.2:3b` runs locally via Ollama and must return JSON:
   `{answer, cited_doc_ids, insufficient_information}`.
4. **Check:** fixed answer rules validate the citations and normalise refusals.
5. **Show:** the CLI (`rag ask`) prints the answer, the cited chunks with their GitHub URLs, and warnings.

## Model choices and why

- **A free, local model via Ollama.** No costs and no API key needed. The model is
  called over Ollama's HTTP API with the standard library, so there is no extra dependency.
- **`llama3.2:3b`.** In a first test of three local models, it was the only one that produced valid JSON, cited exact
  `doc_id`s and abstained correctly on an unanswerable question. It is also the fastest (~30 s per question on this
  CPU), and its 128k context window is larger than needed.
- **Deterministic and reproducible.** Temperature 0, seed 42, and an explicit context window (`num_ctx = 8192`), because
  Ollama silently drops the start of prompts longer than its small default. Before each evaluation run the model is
  unloaded, which clears Ollama's prompt cache. Every run stores the model digest, settings, prompt fingerprint and
  answer-rules version, and a resumed run with different settings is refused.
- **Small context budget: top 4 chunks of at most 2,000 characters, at most 800 output tokens** (see the section on
  the context budget below).
- **Structured JSON output.** The answer, the citations and "I don't know" are machine-readable, and Ollama constrains
  the output to the schema. An explicit `insufficient_information` field makes abstaining a measurable decision.
- **Citations by exact `doc_id`, checked in code.** A citation to a chunk the model was not given is dropped, unless it
  is an obvious copying error that matches exactly one given chunk; then it is repaired. Repairs and dropped
  citations are recorded.
- **The prompt grounds the answer and allows partial answers** (round 3, [third_results.md](third_results.md)). The
  model answers only with information from the sources, answers the part they cover, and refuses when they do not
  contain the answer, even if they are about the same topic. Every statement must be backed by a cited source. An
  earlier prompt that refused only when no source was about the topic made the model answer almost everything
  (round 2); a strict "do not guess" wording made it refuse most real questions.
- **The refusal sentence is enforced in code, not in the prompt.** Every refusal gets the same sentence ("The knowledge
  source does not contain enough information…") and no citations. When the prompt dictated that sentence, the model
  copied it mechanically and refused most answerable questions.
- **Retrieval query and question can differ.** `--query-mode` searches with a shorter part of the question, while the
  model always answers the full question. The default is the full question.
- **One CLI, `rag`.** It has a subcommand per step and imports modules lazily (`rag ask` does not load the dataset
  library). Shared defaults live in `config.py`, so the CLI, the assistant and the evaluation use the same values.

## Models tried

All models run locally via Ollama. As the assistant, each model got
the same two questions (one answerable, one unanswerable) with the same top-4 chunks and JSON schema.

| Model | Role | Advantages | Disadvantages | Outcome |
|---|---|---|---|---|
| **`llama3.2:3b`** (3B, ~2 GB, 128k context) | Assistant | Valid JSON, exact citations and a correct "insufficient" in the test; fastest (25-39 s per question, ~108 tokens/s reading); a large context window | Weak at following instructions on real questions: copies long `doc_id`s wrongly, cites every chunk it gets, and its refusals follow the prompt wording rather than the evidence | **Chosen as the assistant** |
| `qwen2.5:3b` (3B) | Assistant | Correct answer with exact citations on the answerable question; reads prompts slightly faster (~120 tokens/s) | Answered the unanswerable question from its own knowledge, against the instructions; hit the output-token limit, so its JSON was invalid; slower per question overall (41-47 s) | Not chosen |
| `qwen2.5:3b` | Judge | 2.5x faster than `llama3` (~32 s instead of ~79 s per answered question) | Barely agrees with `llama3` on nuggets (kappa 0.18) and fails to produce a usable verdict more often | Not chosen |
| `llama3` (8B, 8k context) | Assistant | Correct answer on the answerable question | Cited a `doc_id` it was never given; slow (61-83 s per question, ~40 tokens/s reading; 3.6 min with longer chunks); only an 8k context window | Not chosen as the assistant |
| **`llama3`** (8B) | Judge | Larger than, and different from, the assistant, so the assistant does not grade its own answers | Slow (~70-80 s per answered question); its verdicts are not verified | **Chosen as the judge** |

**Considered but not tried:** a hosted model such as Claude, or a free tier such as Gemini or Groq. These are
probably much better at exact citations and at refusing, and take seconds per question. I did not use them, to avoid API keys, costs and rate limits.

## Context budget: top-k and chunk size

The two settings decide how much retrieved text the model reads:
- **`top_k`** is the number of chunks the model gets.
- **`max_chars`** is where each chunk is cut, with a `[... truncated]` marker.

Together they set the text budget: `top_k × max_chars`, so 8,000 characters (~2.3k tokens with the question) by default.

**I chose a 4 top_k × 2,000 max_chars setup as the default:**
- **Speed.** Reading the prompt dominates the time on this CPU. At ~2.3k tokens a question takes ~25-40 s, so a run on
  20 questions takes minutes rather than hours. With 5 full-length chunks (~6k tokens), a question took minutes.
- **It fits the context window.** Even the longest question (~13.6k characters) plus 8,000 characters of chunks
  should stay below `num_ctx = 8192` tokens, so Ollama does not silently drop the start of the prompt. The backend
  logs a warning if a prompt does fill the window.
- **Breadth over a single chunk.** BM25 rarely puts the most useful chunk first, so 4 chunks give a better chance that
  some evidence is in the context than 1 or 2.
- **Not too many for a small model.** `llama3.2:3b` tends to cite every chunk it gets. More chunks mean more noise to
  read and more irrelevant citations.
- **2,000 characters is about half a median chunk** (~3.9k). This keeps the start of each chunk, where the file
  header, class or section title usually is, within the budget.
- **800 output tokens** leave room for an answer with code without cutting off the JSON.

**The trade-offs:**

| | Benefit | Risk |
|---|---|---|
| Fewer, longer chunks (low `top_k`, high `max_chars`) | Chunks shown more completely; fewer irrelevant citations | A relevant chunk at rank 3-4 drops out; one bad top hit takes half the context |
| More, shorter chunks (high `top_k`, low `max_chars`) | More documents, so a better chance the evidence is included | Fragments (imports, notebook metadata) without the explanation; more indiscriminate citing |
| Larger total budget | More evidence | Slower; long questions get close to the context window; more noise for a small model |
| Smaller total budget | Faster, better for interactive use | Less evidence; the model refuses or guesses more often |

## Limitations and alternatives

These follow from the design itself, whatever the results turn out to be.

- **Only the model decides whether the context is sufficient.** The prompt allows partial answers and there is no
  relevance gate, so a small model's own judgment is the only check. → *Evidence first:* the model quotes the
  sentences it relies on, and code refuses when no quote is found in the sources. Or an embedding-similarity gate,
  which comes with dense retrieval.
- **Citations are checked to exist, not to support the answer.** A citation to a provided but unrelated chunk passes.
  → Quote verification at answer time, or a citation-support check with a judge.
- **Citation repair can credit a wrong id.** It maps a mangled id to the one provided chunk it resembles, which is not
  necessarily the chunk the model meant. → Short source ids in the prompt (`[1]`-`[4]`), mapped back to `doc_id`s in
  code.
- **Refusal detection by wording is a heuristic.** It only recognises known phrasings in short answers. → Rely on a
  flag with a stronger model, or evidence first.
- **The assistant has no minimum-answer check.** A one-word answer is shown as a normal answer. → The same check the
  evaluation uses for degenerate answers.
- **Chunks are cut at a fixed length,** so the part that answers the question may be cut off. → Add neighbouring
  chunks, or select the most relevant part of a chunk instead of its first 2,000 characters.
