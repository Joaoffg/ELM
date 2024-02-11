## ELM
The Erasmian Language Model

ELM is a community driven large language model tailored to the research and education needs of Erasmus University (EUR, Netherlands) students and staff.

The model draws inspiration from ChatGPT in terms of architecture, but it aims to be privacy senstitive, environmentally constious, and from and for the Erasmus community. Here are a few key points of ELM:

1. The undelying language model is trained and fine-tuned on academic outputs from Erasmus University, such as scientific papers or student theses;
2. Training and fine-tuning the model is a joint effort of students and staff, transparent for all parties involved;
3. The prompt-response examples used to fine tune the model come from students and staff, not crowdsourcing services;
4. Defining what is the "better" model output also comes from the perspective of research and education.

The true richness of ELM lies in the way its training data is generated. What is the "state-of-the-art" model may change quickly, but quality data will maintain its relevance and ensure that ELM and its future iterations serve the needs of the community that nurtured it.

We hope that the ELM experience becomes a template for community driven, decentralized and purpuseful AI development and application.

# Models

ELM_Small Pre-trained - https://surfdrive.surf.nl/files/index.php/s/9EQ0V9XlfbqZJpb/download
A 166M parameter Llama2 based model trained for 3 epochs on the ELM dataset

ELM_Large Pre-trained - https://surfdrive.surf.nl/files/index.php/s/Qe4kqhx3o8BScvu
A 900M parameter Llama2 based model trained for 3 epochs on the ELM data

ELM_Large Chat - See adapter weights in GitHub repo
A 900M parameter Llama2 based model fine-tuned on the ELM fine-tuning data, the Stanford Alpaca data, and a translated Dutch version of Stanford Alpaca (https://huggingface.co/datasets/BramVanroy/alpaca-cleaned-dutch)

# Datasets

A sample of the dataset can be found in the GitHub repository. The pretraining dataset consists of:
- All public MA theses from Erasmus University, accessible here: https://thesis.eur.nl/
- All Erasmus University research outputs, accessible here: https://pure.eur.nl/
- A student generated dataset for chat fine tuning and RLHF: ELM Fine-tune dataset_2023_10_09.xlsx
- The Stanford Alpaca dataset for fine-tuning: https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release
- The Dutch translated Stanford Alpaca dataset for fine-tuning: https://huggingface.co/datasets/BramVanroy/alpaca-cleaned-dutch

Pretraining data was preprocessed to eliminate chunks of text with less than 25 tokens.

# Model training
Training scripts are provided in the GitHub repository. All models were trained for three epochs in the EUR datasets. Fine tuning LoRA parameters are available in the fine-tuning scripts.

We trained our own EUR SentencePiece tokenizer based on the EUR pretraining data.
