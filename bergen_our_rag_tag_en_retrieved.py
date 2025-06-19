#BERGEN
#Copyright (c) 2024-present NAVER Corp.
#CC BY-NC-SA 4.0 license

import hydra
from multiprocess import set_start_method
import os
import json
if 'CONFIG' in  os.environ:
    CONFIG = os.environ["CONFIG"]
else:
    CONFIG= 'rag_our_rag_tag_en_retrieved'

@hydra.main(config_path="config", config_name=CONFIG, version_base="1.2")
def main(config):

    from bergen.modules.rag_our_rag_tag_en_retrieved import RAG
    rag = RAG(**config, config=config)

    if 'train' in config:
        rag.train()
    if 'dataset_split' in config:
        dataset_split = config['dataset_split']
    else:
        dataset_split = 'dev'
    rag.eval(dataset_split=dataset_split)

if __name__ == "__main__":
    # needed for multiprocessing to avoid CUDA forked processes error
    # https://huggingface.co/docs/datasets/main/en/process#multiprocessing
    set_start_method("spawn")
    main()
