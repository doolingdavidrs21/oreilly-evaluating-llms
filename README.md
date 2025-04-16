![oreilly-logo](images/oreilly.png)


go here to check that the ollama server is running: http://127.0.0.1:11434



# Evaluating Large Language Models (LLMs)

This repository contains code for live session [O'Reilly Course on Evaluating LLMs](https://learning.oreilly.com/live-events/evaluating-large-language-models-llms/0642572013878) with companion video course [here](https://learning.oreilly.com/course/evaluating-large-language/9780135451922/)

This course offers an in-depth look at evaluating large language models (LLMs), equipping participants with the tools and techniques to measure their performance, reliability, and task alignment. Topics range from foundational metrics to advanced methods such as probing and fine-tuning evaluation. Hands-on exercises and real-world case studies make this course engaging and practical, ensuring learners can directly apply their knowledge to real-world systems.

## Notebooks

In the activated environment, run

```bash
python3 -m jupyter notebook
```

- **Evaluating Generative Tasks**

	- **[Evaluating Generative Free Text with Rubrics](https://colab.research.google.com/drive/1DeVYrdNb3FlQQLeBqGPFkx6roZaPwVRy?usp=sharing)**

	- **[Perplexlity, SelfCheckGPT + BERTScore](https://colab.research.google.com/drive/1rG8vCJz5He5JM5oPLYnH3TShSyCnyK9H?usp=sharing)** -


- **Evaluating Understanding Tasks**

	- **[Classification Metrics with BERT and BART](https://colab.research.google.com/drive/1yALtgSK6ENEa5WkBGWm3DPviuVGwhrw9?usp=sharing)** - Comparing a fine-tuned BERT model vs 0-shot classification with BART on the [app_reviews dataset](https://huggingface.co/datasets/sealuzh/app_reviews)

		- [Fine-tuning BERT on app_reviews](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/05_bert_app_review.ipynb): Fine-tuning a BERT model for app review classification.

		- [Fine-tuning Openai on app_reviews](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/05_openai_app_review_fine_tuning.ipynb): Fine-tuning OpenAI models for app review classification.


	- **[RAG - Retrieval](https://github.com/sinanuozdemir/oreilly-retrieval-augmented-gen-ai/blob/main/notebooks/RAG_Retrieval.ipynb)**: An introduction to vector databases, embeddings, and retrieval

		- [Advanced Semantic Search](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/02_semantic_search.ipynb): A more advanced notebook on semantic search, cross-encoders, and fine-tuning from my [book](https://github.com/sinanuozdemir/quick-start-guide-to-llms)


- **Benchmarking**

	- **[Benchmarking Llama 3.2 Instruct on MMLU and Embedders on MTEB](https://colab.research.google.com/drive/1zDCqXc7vHoZilHVe3y2lYyTmSUSe6bh3?usp=sharingb)**


		- [Follow-up Evaluating Llama 3.2 non-instruct on MMLU](https://colab.research.google.com/drive/1aMy19Ikyody9CGyn42K3E_DQwLScL0Ek?usp=sharing)

		- [Evaluating Llama 3.1 vs Mistral on Truthful Q/A](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/12_llm_gen_eval.ipynb) -


- **Probing**

	- **[Probing Chess Playing LLMs](https://colab.research.google.com/drive/114turFLNxLJXiIseDWl1BDJmont0VD8h?usp=sharing)**

	- There are over a dozen notebooks for the birth year/death year probing example so I will only share a few key ones here:
	  - [Llama-3 8B Instruct with prompt "Who is {NAME}"](https://colab.research.google.com/drive/1e1d9fATVjVun-_tPj4vS_DSTGaIfxs01?usp=sharing)
	  - [BERT-large-cased no prompt](https://colab.research.google.com/drive/1cizgoh1J6Y-DHBrOkNTFo9Y1CypjwuQM?usp=sharing)
	  - [Mistral-7B-Instruct-v0.3 with prompt "Who is {NAME}"](https://colab.research.google.com/drive/1VL3betxqVZ_H3_8XmLbjE0hEjaoy-HPV?usp=sharing)

- **Evaluating Fine-tuning**

	- **[Optimizing Fine-tuning](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/10_optimizing_fine_tuning.ipynb)** - Best practices for optimizing fine-tuning of transformer models.

	- **Evaluating Fine-tuning Data**

		- **[AUM + Cosine Similarity to clean data](https://colab.research.google.com/drive/1hPnU9sLsV9W50q9rd_oxUU1Bv7SUCVU5?usp=sharing)**

- **Case Studies**

	- **[Evaluating AI Agents: Task Automation and Tool Integration](https://ai-office-hours.beehiiv.com/p/evaluating-ai-agent-tool-selection)**
		- [Positional Bias on Agent Response Evaluation](https://github.com/sinanuozdemir/oreilly-ai-agents/blob/main/notebooks/Evaluating_LLMs_with_Rubrics.ipynb)

	- **[Measuring RAG Re-Ranking](https://ai-office-hours.beehiiv.com/p/re-ranking-rag)**

	- **[Building and Evaluating a Recommendation Engine Using LLMs](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/07_recommendation_engine.ipynb)** - Fine-tuning embedding engines using custom preference data

	- **[Using Evaluation to combat AI drift](https://colab.research.google.com/drive/14E6DMP_RGctUPqjI6VMa8EFlggXR7fat?usp=sharing)**

	- **[Time Series Regression](https://colab.research.google.com/drive/1VRB1774lq5s0loxDpDXGTw5qAF9FUseH?usp=sharing)** - Predicting the price of Bitcoin



## Instructor

**Sinan Ozdemir** is the Founder and CTO of LoopGenius where he uses State of the art AI to help people run digital ads on Meta, Google, and more. Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a master’s degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.

