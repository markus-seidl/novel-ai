import os
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from summarizer.sbert import SBertSummarizer

MODEL: SBertSummarizer = None

PERFORMANCE = {
    "processed_chars": 0,
    "total_time": 0  # seconds
}


def performance_info() -> str:
    chars_per_second = PERFORMANCE["total_time"] / PERFORMANCE["processed_chars"]
    return "{:.7f} s/char".format(chars_per_second)


def ensure_model():
    global MODEL
    if MODEL is None:
        MODEL = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        print("Initialized SBERT with: ", MODEL.model.device)


def summarize_text(text, length) -> str:
    ensure_model()
    ratio = min(length / len(text), 0.95)

    factor = 0.95
    for i in range(0, 3):
        len_before = len(text)
        PERFORMANCE['processed_chars'] += len_before

        start_time = time.time()
        result = MODEL(
            text, min_length=20, max_length=300, ratio=ratio
        )
        PERFORMANCE['total_time'] += (time.time() - start_time)

        if len(result) > length * 1.1:
            ratio = (length * factor) / len(text)
            factor -= 0.1
        else:
            return result

        if ratio < 0.000001:
            break

    return result


if __name__ == '__main__':
    with open("", "r") as f:
        text = f.read()

    summary = summarize_text(text, 500)
    print(len(summary))
    print(summary)
    print(performance_info())
