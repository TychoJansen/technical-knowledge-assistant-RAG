# bm25_word_stem: 5 test questions with the lowest Recall@50

## Databricks Dolly LLM: empty result when using LangChain with context

- qid: 76124316 | language: python | tags: databricks, huggingface-transformers, langchain, large-language-model
- question length: 827 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.70

Top-5 retrieved:

1. `langchain/libs/langchain/tests/unit_tests/evaluation/qa/test_eval_chain.py_0_4998` (non-relevant, score 60.7)
2. `langchainjs/docs/core_docs/docs/how_to/few_shot_examples.ipynb_0_7631` (unjudged, score 52.0)
3. `langchainjs/examples/src/chains/retrieval_qa_sources.ts_0_7078` (non-relevant, score 51.0)
4. `langchain/docs/docs/how_to/few_shot_examples.ipynb_0_7622` (unjudged, score 49.6)
5. `langchainjs/docs/core_docs/docs/integrations/retrievers/exa.ipynb_24291_30454` (unjudged, score 49.5)

Missed relevant docs (not in top-50), 3 of 3:

- `transformers/src/transformers/pipelines/text_generation.py_10266_19444`
- `transformers/src/transformers/pipelines/text_generation.py_1147_10260`
- `transformers/tests/pipelines/test_pipelines_text_generation.py_10758_19047`

## Get all documents from ChromaDb using Python and langchain

- qid: 76184540 | language: python | tags: langchain, py-langchain, chromadb
- question length: 863 chars
- R@50: 0.00 | R@100: 0.25 | Cov@20: 0.00 | Unj@10: 0.60

Top-5 retrieved:

1. `langchain/docs/docs/how_to/vectorstores.mdx_0_5549` (non-relevant, score 88.7)
2. `langchainjs/docs/core_docs/docs/integrations/vectorstores/chroma.ipynb_0_6875` (non-relevant, score 81.5)
3. `langchainjs/docs/core_docs/docs/integrations/retrievers/self_query/chroma.ipynb_0_6971` (unjudged, score 81.2)
4. `llama_index/docs/docs/understanding/storing/storing.md_0_4924` (unjudged, score 80.4)
5. `langchain/docs/docs/integrations/retrievers/self_query/chroma_self_query.ipynb_0_6766` (unjudged, score 78.3)

Missed relevant docs (not in top-50), 4 of 4:

- `chroma/clients/js/test/get.collection.test.ts_0_4481`
- `chroma/docs/docs.trychroma.com/pages/guides/index.md_15395_22747`
- `chroma/docs/docs.trychroma.com/pages/reference/cheatsheet.md_0_6165` (rank 85)
- `langchain/libs/partners/chroma/langchain_chroma/vectorstores.py_31944_40606`

## Is there a way to stream output in Fastapi from the response I get from llama-index

- qid: 76288582 | language: python | tags: python, langchain, llama-index
- question length: 3393 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.70

Top-5 retrieved:

1. `llama_index/llama-index-legacy/llama_index/legacy/service_context.py_8930_15528` (non-relevant, score 241.6)
2. `llama_index/llama-index-legacy/llama_index/legacy/service_context.py_0_8924` (non-relevant, score 231.9)
3. `llama_index/llama-index-legacy/tests/indices/response/test_response_builder.py_0_7367` (non-relevant, score 200.0)
4. `llama_index/experimental/cli/configuration.py_0_3969` (unjudged, score 185.6)
5. `llama_index/llama-index-legacy/tests/indices/response/test_response_builder.py_7370_11145` (unjudged, score 180.2)

Missed relevant docs (not in top-50), 12 of 12:

- `langchain/libs/community/langchain_community/llms/oci_data_science_model_deployment_endpoint.py_20509_29561`
- `langchainjs/docs/core_docs/docs/how_to/streaming_llm.mdx_0_2156`
- `langchainjs/examples/src/prompts/http_response_output_parser_custom.ts_0_1325`
- `llama_index/docs/docs/examples/workflow/router_query_engine.ipynb_0_7747`
- `llama_index/llama-index-core/llama_index/core/base/response/schema.py_0_8223`
- `llama_index/llama-index-integrations/llms/llama-index-llms-maritalk/llama_index/llms/maritalk/base.py_10455_13373`
- `llama_index/llama-index-integrations/llms/llama-index-llms-perplexity/llama_index/llms/perplexity/base.py_8256_17020`
- `llama_index/llama-index-integrations/readers/llama-index-readers-sec-filings/llama_index/readers/sec_filings/prepline_sec_filings/api/section.py_7363_13785`
- `llama_index/llama-index-legacy/llama_index/legacy/chat_engine/types.py_0_7676`
- `llama_index/llama-index-legacy/llama_index/legacy/llms/perplexity.py_8907_15005`
- `openai-cookbook/examples/data/oai_docs/overview-with-streaming.txt_0_3317`
- `transformers/docs/source/en/pipeline_webserver.md_0_6461`

## Why am I getting an authentication error when trying to run a LangChain tutorial on FAISS vector database with OpenAI API?

- qid: 76322025 | language: python | tags: python, openai-api, langchain, faiss, vector-database
- question length: 608 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.90

Top-5 retrieved:

1. `langchain/docs/docs/contributing/documentation/style_guide.mdx_0_9160` (unjudged, score 43.7)
2. `langchain/docs/docs/integrations/vectorstores/yellowbrick.ipynb_0_7111` (non-relevant, score 43.7)
3. `llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-mongodb/README.md_0_5949` (unjudged, score 43.4)
4. `langchain/docs/docs/integrations/memory/google_alloydb.ipynb_0_7719` (unjudged, score 42.0)
5. `langchain/docs/docs/integrations/vectorstores/google_alloydb.ipynb_0_6839` (unjudged, score 41.6)

Missed relevant docs (not in top-50), 21 of 21:

- `azure-openai/Basic_Samples/AAD_Integration/aad_integration_example_restapi.ipynb_0_6979`
- `azure-openai/Basic_Samples/Chat/chat_with_your_own_data.ipynb_0_7985`
- `azure-openai/Basic_Samples/LangChain/README.md_0_2289`
- `langchain/docs/docs/integrations/chat/azure_chat_openai.ipynb_0_6088`
- `langchain/docs/docs/integrations/llms/azure_openai.ipynb_0_7303`
- `langchain/docs/docs/troubleshooting/errors/MODEL_AUTHENTICATION.mdx_0_754`
- `langchain/libs/community/langchain_community/embeddings/azure_openai.py_0_7435`
- `langchain/libs/community/langchain_community/embeddings/openai.py_4621_11439`
- `langchain/libs/partners/openai/langchain_openai/embeddings/azure.py_0_8542`
- `langchainjs/docs/core_docs/docs/troubleshooting/errors/MODEL_AUTHENTICATION.mdx_0_789`
- `langchainjs/examples/src/guides/deployment/error_handling.ts_0_1465`
- `langchainjs/examples/src/guides/deployment/stream_error_handling.ts_0_1466`
- `llama_index/docs/docs/examples/customization/llms/AzureOpenAI.ipynb_0_6749`
- `llama_index/docs/docs/examples/llm/azure_openai.ipynb_0_6223`
- `llama_index/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/utils.py_0_3235`
- `llama_index/llama-index-legacy/llama_index/legacy/llms/litellm_utils.py_0_6636`
- `openai-cookbook/examples/azure/archive/chat_with_your_own_data.ipynb_0_7894`
- `openai-cookbook/examples/azure/archive/functions.ipynb_0_7712`
- `openai-cookbook/examples/data/oai_docs/error-codes.txt_0_8377`
- `openai-cookbook/examples/data/oai_docs/error-codes.txt_17439_22218`
- `openai-cookbook/examples/data/oai_docs/error-codes.txt_8382_17433`

## Trying to do a docker build which fails at chromadb installation

- qid: 76341201 | language: python | tags: python-3.x, docker, langchain, duckdb, chromadb
- question length: 6461 chars
- R@50: 0.00 | R@100: 0.00 | Cov@20: 0.00 | Unj@10: 0.70

Top-5 retrieved:

1. `llama_index/docs/docs/examples/cookbooks/oreilly_course_cookbooks/Module-8/Advanced_RAG_with_LlamaParse.ipynb_39100_44655` (non-relevant, score 511.3)
2. `langchain/cookbook/fireworks_rag.ipynb_5606_11528` (unjudged, score 507.9)
3. `langchain/docs/docs/integrations/llms/replicate.ipynb_0_6614` (unjudged, score 479.4)
4. `langchain/docs/docs/integrations/tools/e2b_data_analysis.ipynb_7164_13871` (non-relevant, score 478.3)
5. `llama_index/docs/docs/examples/retrievers/vectara_auto_retriever.ipynb_16493_21483` (unjudged, score 476.2)

Missed relevant docs (not in top-50), 2 of 2:

- `chroma/docs/docs.trychroma.com/pages/troubleshooting.md_0_3352`
- `chroma/rust/index/build.rs_0_486`
