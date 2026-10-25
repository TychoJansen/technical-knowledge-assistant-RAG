# bm25_word_stem_q-title_first_paragraph: 5 dev questions with the lowest Recall@50

## ChromaDb add single document, only if it doesn't exist

- qid: 76265631 | language: python | tags: python, langchain, chromadb
- question length: 817 chars
- R@50: 0.00 | R@100: 0.20 | Cov@20: 0.00 | Unj@10: 0.40

Top-5 retrieved:

1. `langchain/libs/partners/chroma/langchain_chroma/vectorstores.py_4107_13535` (non-relevant, score 30.1)
2. `langchainjs/docs/core_docs/docs/integrations/vectorstores/chroma.ipynb_0_6875` (non-relevant, score 28.5)
3. `langchain/libs/community/langchain_community/vectorstores/chroma.py_0_9468` (non-relevant, score 28.1)
4. `chroma/docs/docs.trychroma.com/pages/guides/multimodal.md_0_4479` (unjudged, score 27.9)
5. `langchainjs/libs/langchain-community/src/vectorstores/chroma.ts_0_4653` (unjudged, score 27.4)

Missed relevant docs (not in top-50), 5 of 5:

- `langchain/docs/docs/concepts.mdx_33170_41902`
- `langchain/libs/core/langchain_core/indexing/api.py_0_6186`
- `langchain/libs/partners/chroma/tests/integration_tests/test_vectorstores.py_7267_15685` (rank 55)
- `llama_index/llama-index-core/llama_index/core/ingestion/pipeline.py_14041_22080`
- `transformers/examples/research_projects/codeparrot/scripts/preprocessing.py_0_7440`

## How can I stream a response from LangChain's OpenAI using Flask API?

- qid: 76284412 | language: python | tags: python, flask, openai-api, langchain
- question length: 2496 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.90

Top-5 retrieved:

1. `chroma/examples/chat_with_your_documents/main.py_0_4530` (non-relevant, score 60.1)
2. `llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py_3305_11645` (unjudged, score 55.5)
3. `langchain/libs/community/langchain_community/chat_models/openai.py_5639_14465` (unjudged, score 55.3)
4. `langchain/libs/community/langchain_community/llms/openai.py_4286_12195` (unjudged, score 53.1)
5. `langchain/libs/community/langchain_community/chat_models/konko.py_980_9994` (unjudged, score 52.8)

Missed relevant docs (not in top-50), 7 of 7:

- `langchain/docs/docs/how_to/custom_callbacks.ipynb_0_4991`
- `langchain/libs/community/langchain_community/chat_models/sparkllm.py_13646_22738`
- `langchain/libs/community/langchain_community/llms/textgen.py_7966_14374`
- `langchainjs/environment_tests/test-exports-vercel/src/pages/api/hello-edge.ts_0_2180`
- `llama_index/llama-index-core/llama_index/core/langchain_helpers/streaming.py_0_1352`
- `llama_index/llama-index-legacy/llama_index/legacy/langchain_helpers/streaming.py_0_1354`
- `transformers/docs/source/en/pipeline_webserver.md_0_6461`

## langchain pipeline vram usage when loading model

- qid: 76293427 | language: python | tags: python, pytorch, huggingface-transformers, langchain
- question length: 7279 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.60

Top-5 retrieved:

1. `langchain/docs/docs/integrations/providers/rwkv.mdx_0_1947` (non-relevant, score 29.2)
2. `transformers/docs/source/en/llm_tutorial_optimization.md_3315_11072` (non-relevant, score 28.3)
3. `transformers/docs/source/en/llm_tutorial_optimization.md_11073_17515` (non-relevant, score 22.4)
4. `transformers/examples/tensorflow/language-modeling-tpu/README.md_0_6237` (unjudged, score 19.1)
5. `transformers/docs/source/en/quicktour.md_1644_10151` (unjudged, score 18.5)

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

## map_reduce not working as expected using langchain

- qid: 76456041 | language: python | tags: openai-api, chatgpt-api, langchain
- question length: 2142 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 1.00

Top-5 retrieved:

1. `langchainjs/docs/core_docs/docs/how_to/graph_prompting.ipynb_0_7580` (unjudged, score 35.9)
2. `llama_index/llama-index-integrations/llms/llama-index-llms-vllm/tests/cassettes/TestVllmIntegration.test_astream_completion.yaml_344910_354023` (unjudged, score 34.8)
3. `llama_index/llama-index-integrations/llms/llama-index-llms-vllm/tests/cassettes/TestVllmIntegration.test_astream_completion.yaml_253318_262509` (unjudged, score 34.5)
4. `llama_index/llama-index-integrations/llms/llama-index-llms-vllm/tests/cassettes/TestVllmIntegration.test_astream_completion.yaml_372475_381715` (unjudged, score 34.3)
5. `llama_index/llama-index-integrations/llms/llama-index-llms-vllm/tests/cassettes/TestVllmIntegration.test_astream_completion.yaml_280757_290083` (unjudged, score 34.1)

Missed relevant docs (not in top-50), 16 of 16:

- `langchain/docs/api_reference/guide_imports.json_143910_146795`
- `langchain/docs/api_reference/guide_imports.json_86915_94579`
- `langchain/docs/docs/how_to/summarize_map_reduce.ipynb_0_7085`
- `langchain/docs/docs/integrations/graphs/neo4j_cypher.ipynb_12985_19096`
- `langchain/docs/docs/tutorials/summarization.ipynb_0_7359`
- `langchain/docs/docs/tutorials/summarization.ipynb_14440_21723`
- `langchain/docs/docs/versions/migrating_chains/map_reduce_chain.ipynb_0_7362`
- `langchain/docs/docs/versions/migrating_chains/stuff_docs_chain.ipynb_0_7018`
- `langchain/libs/langchain/langchain/chains/combine_documents/map_reduce.py_10642_12129`
- `langchainjs/examples/src/chains/question_answering.ts_0_1034`
- `langchainjs/examples/src/guides/fallbacks/long_inputs.ts_0_693`
- `langchainjs/langchain/src/chains/question_answering/tests/load.int.test.ts_0_1818`
- `langchainjs/langchain/src/chains/tests/combine_docs_chain.test.ts_0_2927`
- `llama_index/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/paged_csv/README.md_0_882`
- `openai-cookbook/examples/data/example_pdfs/models-page.pdf_6602_13018`
- `openai-cookbook/examples/data/parsed_pdf_docs.json_35595_42458`

## How to get more detailed results sources with Langchain

- qid: 76482024 | language: python | tags: python, openai-api, gpt-3, langchain, chatgpt-api
- question length: 1180 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.60

Top-5 retrieved:

1. `langchain/docs/docs/integrations/document_loaders/diffbot.ipynb_0_7651` (unjudged, score 59.4)
2. `openai-cookbook/examples/chatgpt/gpt_actions_library/gpt_action_sharepoint_text.ipynb_24323_32471` (non-relevant, score 58.9)
3. `llama_index/docs/docs/examples/vector_stores/WordLiftDemo.ipynb_21607_30041` (unjudged, score 58.7)
4. `openai-cookbook/examples/vector_databases/pinecone/GPT4_Retrieval_Augmentation.ipynb_60772_69932` (non-relevant, score 58.3)
5. `openai-cookbook/examples/vector_databases/pinecone/GPT4_Retrieval_Augmentation.ipynb_69933_78560` (non-relevant, score 57.8)

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
