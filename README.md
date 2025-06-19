## This Repository is built upon BERGEN: A Benchmarking Library for Retrieval-Augmented Generation
 
[![arXiv](https://img.shields.io/badge/arXiv-2407.01102-b31b1b.svg)](https://arxiv.org/abs/2407.01102)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Key Features

- Baseline
- CROSSRAG
- DKM-RAG
- HARD-FILTER
- QTT-RAG


For more information and experimental findings, please see baseline papers:
- CROSSRAG paper: https://arxiv.org/abs/2504.03616 
- DKM-RAG paper: https://arxiv.org/abs/2502.11175 

## Quick Start

You can configure each component using simple YAML files. Here's an example of running an experiment:
- This is example script for HARD-FILTER, QTT-RAG.
```bash
RUN_NAME="Your RUN_NAME"
EXP_FOLDER="Your EXP_FOLDER"
LOG_FOLDER="Your LOG_FOLDER"

python bergen_our_rag_tag.py \
  run_name="$RUN_NAME" \
  generator='exaone-3.5-7.8b' \
  translation_model='llama-31-8b-instruct' \ ### DKM-RAG needs refine_model instead of translation_model, No need to specify neither of them for CROSSRAG and Baseline  ###
  retriever='bge-m3' \
  reranker='bge-m3' \
  dataset='xor_tydiqa/xor_tydiqa_ar.retrieve_en_ar.yaml' \
  prompt='basic_translated_langspec/ar_score' ++experiments_folder="$EXP_FOLDER" ++run_name="$RUN_NAME" ++log_folder="$LOG_FOLDER" 
```
Change bergen files, configs and corresponding languages as you needed.

## Installation

Check : https://github.com/naver/bergen/blob/main/documentation/INSTALL.md 

