import sys
from datasets import Dataset

sys.path.insert(1, '../dataset/')

import loader_llm_21

if __name__ == '__main__':
    local_temp = "./temp/dl/"

    ds: Dataset = loader_llm_21.load_novel_dataset(local_temp, None)
    ds.save_to_disk("./dataset")
