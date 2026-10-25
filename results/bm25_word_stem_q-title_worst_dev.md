# bm25_word_stem_q-title: 5 dev questions with the lowest Recall@50

## Do LOWER results from Chroma's similarity_with_score mean HIGHER Accuracy?

- qid: 76080831 | language: python | tags: langchain, vector-database, chromadb
- question length: 895 chars
- R@50: 0.00 | R@100: 0.08 | Cov@20: 0.00 | Unj@10: 1.00

Top-5 retrieved:

1. `langchainjs/docs/core_docs/docs/how_to/reduce_retrieval_latency.mdx_0_2367` (unjudged, score 10.3)
2. `llama_index/llama-index-packs/llama-index-packs-koda-retriever/examples/data/llm_compiler.pdf_58675_62122` (unjudged, score 9.8)
3. `chroma/examples/deployments/do-terraform/variables.tf_0_3122` (unjudged, score 8.6)
4. `azure-openai/End_to_end_Solutions/AOAIVirtualAssistant/src/botapp/cognition/inputdata.yml_0_2681` (unjudged, score 8.3)
5. `chroma/docs/docs.trychroma.com/pages/troubleshooting.md_0_3352` (unjudged, score 8.2)

Missed relevant docs (not in top-50), 13 of 13:

- `chroma/rust/distance/benches/distance_metrics.rs_0_696`
- `langchain/docs/docs/integrations/vectorstores/myscale.ipynb_6970_10920`
- `langchain/docs/docs/tutorials/retrievers.ipynb_7325_14954`
- `langchain/libs/community/langchain_community/vectorstores/chroma.py_9474_18197`
- `langchain/libs/community/langchain_community/vectorstores/couchbase.py_8262_17934`
- `langchain/libs/community/langchain_community/vectorstores/faiss.py_10557_19445`
- `langchain/libs/community/langchain_community/vectorstores/qdrant.py_11294_21489`
- `langchain/libs/community/langchain_community/vectorstores/qdrant.py_21495_32111`
- `langchain/libs/partners/chroma/langchain_chroma/vectorstores.py_22343_31938` (rank 85)
- `langchain/libs/partners/qdrant/langchain_qdrant/vectorstores.py_10780_20927`
- `langchain/libs/partners/qdrant/langchain_qdrant/vectorstores.py_20933_31437`
- `langchainjs/langchain/src/retrievers/matryoshka_retriever.ts_0_7353`
- `llama_index/llama-index-legacy/tests/vector_stores/test_lancedb.py_0_1820`

## How can I stream a response from LangChain's OpenAI using Flask API?

- qid: 76284412 | language: python | tags: python, flask, openai-api, langchain
- question length: 2496 chars
- R@50: 0.00 | R@100: 0.14 | Cov@20: 0.00 | Unj@10: 0.80

Top-5 retrieved:

1. `llama_index/docs/docs/understanding/putting_it_all_together/apps/fullstack_app_guide.md_0_4389` (non-relevant, score 11.4)
2. `llama_index/docs/docs/understanding/putting_it_all_together/apps/fullstack_app_guide.md_11627_16422` (unjudged, score 10.4)
3. `llama_index/docs/docs/understanding/putting_it_all_together/apps/fullstack_app_guide.md_4389_11627` (unjudged, score 10.4)
4. `langchain/docs/docs/integrations/memory/redis_chat_message_history.ipynb_0_7370` (unjudged, score 10.2)
5. `langchainjs/docs/core_docs/docs/how_to/chat_token_usage_tracking.mdx_0_3551` (unjudged, score 10.1)

Missed relevant docs (not in top-50), 7 of 7:

- `langchain/docs/docs/how_to/custom_callbacks.ipynb_0_4991`
- `langchain/libs/community/langchain_community/chat_models/sparkllm.py_13646_22738`
- `langchain/libs/community/langchain_community/llms/textgen.py_7966_14374`
- `langchainjs/environment_tests/test-exports-vercel/src/pages/api/hello-edge.ts_0_2180` (rank 91)
- `llama_index/llama-index-core/llama_index/core/langchain_helpers/streaming.py_0_1352`
- `llama_index/llama-index-legacy/llama_index/legacy/langchain_helpers/streaming.py_0_1354`
- `transformers/docs/source/en/pipeline_webserver.md_0_6461`

## langchain pipeline vram usage when loading model

- qid: 76293427 | language: python | tags: python, pytorch, huggingface-transformers, langchain
- question length: 7279 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.60

Top-5 retrieved:

1. `transformers/docs/source/en/llm_tutorial_optimization.md_3315_11072` (non-relevant, score 12.0)
2. `transformers/docs/source/en/llm_tutorial_optimization.md_11073_17515` (non-relevant, score 10.1)
3. `langchain/docs/docs/integrations/providers/rwkv.mdx_0_1947` (non-relevant, score 8.6)
4. `llama_index/llama-index-integrations/readers/llama-index-readers-pdf-marker/llama_index/readers/pdf_marker/base.py_0_1901` (unjudged, score 7.3)
5. `langchain/docs/docs/integrations/llms/llamacpp.ipynb_14088_21180` (unjudged, score 6.8)

Missed relevant docs (not in top-50), 10 of 10:

- `transformers/docs/source/en/conversations.md_6746_12294`
- `transformers/docs/source/en/llm_tutorial.md_0_6773`
- `transformers/docs/source/en/model_doc/chameleon.md_756_9559`
- `transformers/docs/source/en/quantization/gptq.md_0_6590`
- `transformers/docs/source/ja/main_classes/quantization.md_0_2600`
- `transformers/docs/source/ko/quantization/gptq.md_756_4640`
- `transformers/docs/source/zh/main_classes/quantization.md_3107_6066`
- `transformers/src/transformers/utils/quantization_config.py_22422_29367`
- `transformers/tests/models/mllama/test_modeling_mllama.py_22805_25262`
- `transformers/tests/quantization/bnb/test_4bit.py_0_3749`

## Vector Stores storage in LangChain

- qid: 76404464 | language: js | tags: javascript, python, langchain
- question length: 448 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.90

Top-5 retrieved:

1. `langchainjs/docs/core_docs/docs/integrations/platforms/microsoft.mdx_0_8527` (unjudged, score 7.7)
2. `llama_index/docs/docs/module_guides/storing/index.md_0_3047` (unjudged, score 7.4)
3. `llama_index/docs/docs/module_guides/storing/customization.md_0_4724` (unjudged, score 7.3)
4. `langchainjs/docs/core_docs/docs/integrations/stores/cassandra_storage.mdx_0_2009` (unjudged, score 7.2)
5. `langchain/docs/docs/integrations/vectorstores/activeloop_deeplake.ipynb_0_7469` (unjudged, score 7.1)

Missed relevant docs (not in top-50), 10 of 10:

- `langchain/docs/docs/integrations/retrievers/docarray_retriever.ipynb_0_6405`
- `langchainjs/docs/core_docs/docs/integrations/retrievers/self_query/hnswlib.ipynb_0_6903`
- `langchainjs/docs/core_docs/docs/integrations/vectorstores/hnswlib.ipynb_0_6756`
- `langchainjs/examples/src/indexes/vector_stores/closevector_saveload.ts_0_1227`
- `langchainjs/examples/src/indexes/vector_stores/faiss_saveload.ts_0_769`
- `langchainjs/examples/src/indexes/vector_stores/hnswlib_saveload.ts_0_756`
- `langchainjs/libs/langchain-community/src/structured_query/tests/hnswlib_self_query.int.test.ts_0_5830`
- `langchainjs/libs/langchain-community/src/vectorstores/hnswlib.ts_1141_9179`
- `langchainjs/libs/langchain-community/src/vectorstores/hnswlib.ts_9182_12395`
- `langchainjs/libs/langchain-community/src/vectorstores/tests/hnswlib.int.test.ts_0_3377`

## How to get more detailed results sources with Langchain

- qid: 76482024 | language: python | tags: python, openai-api, gpt-3, langchain, chatgpt-api
- question length: 1180 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.90

Top-5 retrieved:

1. `langchain/docs/docs/versions/migrating_memory/conversation_buffer_window_memory.ipynb_22589_29633` (unjudged, score 7.8)
2. `langchainjs/docs/core_docs/docs/versions/migrating_memory/conversation_buffer_window_memory.ipynb_21299_24768` (unjudged, score 7.7)
3. `openai-cookbook/examples/chatgpt/gpt_actions_library/gpt_action_sharepoint_text.ipynb_24323_32471` (non-relevant, score 7.7)
4. `langchain/docs/docs/versions/migrating_chains/stuff_docs_chain.ipynb_7019_8906` (unjudged, score 7.6)
5. `langchain/docs/docs/integrations/vectorstores/couchbase.ipynb_20182_26234` (unjudged, score 7.5)

Missed relevant docs (not in top-50), 12 of 12:

- `langchain/docs/docs/concepts.mdx_60728_69794`
- `langchain/docs/docs/how_to/structured_output.ipynb_21155_27810`
- `langchain/libs/langchain/langchain/chains/openai_functions/qa_with_structure.py_0_4841`
- `langchain/libs/langchain/langchain/chains/qa_generation/prompt.py_0_1875`
- `langchain/libs/langchain/langchain/chains/question_answering/stuff_prompt.py_0_1146`
- `langchain/libs/langchain/langchain/chains/retrieval_qa/prompt.py_0_399`
- `langchain/templates/self-query-qdrant/self_query_qdrant/prompts.py_0_555`
- `langchainjs/docs/core_docs/docs/concepts.mdx_57248_66689`
- `langchainjs/docs/core_docs/docs/how_to/structured_output.ipynb_13176_19938`
- `langchainjs/examples/src/prompts/structured_parser.ts_0_2661`
- `langchainjs/examples/src/prompts/structured_parser_sequence.ts_0_2450`
- `langchainjs/examples/src/prompts/structured_parser_zod_sequence.ts_0_2665`
