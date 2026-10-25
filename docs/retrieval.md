# Retrieval: tokenizer and BM25

## The model

**Retriever: BM25** (`bm25s`, Lucene variant, default `k1 = 1.5`, `b = 0.75`). It ranks chunks by the question words
they share, weighting rare words more heavily and correcting for chunk length.

**Tokenizers:**
- `word_stem` (default): lowercased words, Lucene's 33-word stopword list removed, Snowball English stemming.
- `identifier_split`: the same, but it also splits code identifiers (`RecursiveCharacterTextSplitter` → recursive,
  character, text, splitter) and keeps the whole identifier as an extra term.

**Why BM25 with these tokenizers:**
- **Strong, standard baseline.** Technical questions are full of exact identifiers, error messages and API names,
  which is where lexical matching is strong. Default parameters and Lucene's stopwords keep it comparable to a
  standard search engine.
- **Short stopword list.** Longer lists also remove words such as `from` and `after`, which carry meaning in code
  questions.
- **Stemming** lets word forms match (`chains`/`chaining` → `chain`).
- **Cheap and transparent.** The index builds in ~20 s, a search takes ~3 ms, it needs no model or GPU, and you can see
  which terms matched.
- **Build once, search often.** The index is stored on disk per tokenizer, together with the tokenizer name, so
  queries are always tokenized the same way as the corpus.
- **One `Retriever` interface** (`search(query, k)`). Dense, hybrid or reranking retrievers can replace BM25 without
  changing the evaluation or the assistant.
- **Per-word cache in the tokenizer.** About 97% of word occurrences are repeats, so each distinct word is processed
  only once.

## Limitations and alternatives

| Limitation | Alternative |
|---|---|
| **Vocabulary mismatch:** users describe symptoms ("stream a response"), docs use other terms (callbacks). This is the main failure in the error analysis | Dense retrieval with a long-context embedding model, combined with BM25 through Reciprocal Rank Fusion (hybrid) |
| **Long, noisy questions:** pasted logs and package lists match unrelated files (`requirements.txt`) | LLM query rewriting or HyDE. Shorter queries (title, title + first paragraph) can also be tested |
| **Weak ranking at the top:** nDCG@10 is 0.19 | Cross-encoder reranker on the top 50-100. Its ceiling is the first stage's recall (R@100 = 0.41) |
| **Noisy corpus:** raw notebook JSON, `BUILD`/lock files, duplicates | Filter never-relevant file types, extract notebook cell sources, collapse duplicates in the results |
| **The file path is not indexed,** although path words (`document_loaders`, `text_splitters`) are a strong signal | Add the path to the indexed text (index only; the `doc_id` stays the same) |
| **Stemming quirks:** `embed` → `emb` but `embedding` → `embed`; irregular forms are not matched | No stemming or lemmatization; a small effect is expected |
| **Default `k1`/`b`, not tuned,** although chunk lengths range from 1 to 21.8k characters | A small grid on dev, or BM25L/BM25+. Risk of overfitting on 60 questions |
| **Scores depend on the question,** so there is no usable relevance threshold | Use ranks (RRF) for fusion, and an LLM or reranker to decide whether the context is sufficient |
| **A stale index is not detected** | Store a corpus checksum and a tokenizer version in `meta.json`, and refuse to load on a mismatch |

## Results (dev, 60 questions)

| Run | R@10 | R@50 | R@100 | nDCG@10 | MRR@10 | Cov@20 | Unj@10 | Latency |
|---|---|---|---|---|---|---|---|---|
| `word_stem` (baseline) | 0.167 | 0.317 | 0.406 | 0.186 | 0.364 | **0.527** | 0.447 | 3.0 ms |
| `identifier_split` | 0.156 | 0.322 | 0.411 | 0.194 | 0.386 | 0.497 | 0.480 | 4.2 ms |
| `word_stem`, query = title | 0.090 | 0.211 | 0.264 | 0.110 | 0.231 | 0.404 | 0.727 | 0.8 ms |
| `word_stem`, query = title + first paragraph | 0.140 | 0.260 | 0.332 | 0.184 | 0.406 | 0.474 | 0.590 | 1.4 ms |

**What this means:**
- **BM25 finds evidence for about half of the key facts in the top 20** (Cov@20 0.53), but **only 17% of the relevant
  chunks are in the top 10** and 41% in the top 100. Most relevant chunks are never found, so a reranker alone cannot
  fix this.
- **The baseline is in line with the FreshStack paper** (BM25 on LangChain: Cov@20 0.475, R@50 0.261, all 203
  questions). Ours is slightly higher on dev, which suggests the setup is sound.
- **The tokenizers perform about the same.** The differences (±0.01-0.03) are within the noise of 60 questions.
  `word_stem` stays the baseline: it is not worse on the primary metric, and it is cheaper.
- **Shorter queries do not help.** The title alone is clearly worse. Title + first paragraph is about equal at the top
  but finds less further down. Part of the drop may be label bias: shorter queries return many more unjudged chunks
  (Unj@10 up to 0.73), and FreshStack built its judgments with full-question queries.
- **Recall is a lower bound.** 45% of the top 10 was never judged, so the true precision is unknown and probably
  higher.
- **The failures are vocabulary mismatch and noisy queries,** not tokenization. Example: a streaming question
  retrieves generic retrieval-chain code instead of callback docs, and a 7k-character question about VRAM usage
  retrieves `requirements.txt` files.
- **Impact on the answers:** the model sees only the top 4 chunks. Their context coverage stays around 0.25 whatever
  the top-k or chunk length, so retrieval is the bottleneck for answer quality. The next step could be dense or hybrid
  retrieval.
- **On the test split** (143 questions, run once with `word_stem`) the numbers are about the same: Cov@20 0.53,
  R@100 0.37, nDCG@10 0.22 ([third_results.md](third_results.md)).
- **Per-language numbers** (4 JS questions in dev) are not interpretable.
