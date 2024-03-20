from summarizer import Summarizer

MODEL: Summarizer = None


def ensure_model():
    global MODEL
    if MODEL is None:
        MODEL = Summarizer()


def summarize_text(text, length) -> str:
    ensure_model()
    result = MODEL(
        text, min_length=int(0.2 * length), max_length=length
    )
    return result


if __name__ == '__main__':
    with open("../../anna/converted/0a0bf519400a1cc245a4fcbbe6bdd585___Cuckolded___Selvaggio_Tinto.txt", "r") as f:
        text = f.read()

    summary = summarize_text(text, 500)
    print(summary)
