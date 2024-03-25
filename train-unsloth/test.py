from datasets.arrow_writer import SchemaInferenceError
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl.trainer import ConstantLengthDataset


if __name__ == "__main__":
    dataset = Dataset.load_from_disk("./tokenized")
