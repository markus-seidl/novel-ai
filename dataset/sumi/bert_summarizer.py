from summarizer.sbert import SBertSummarizer

model = SBertSummarizer('paraphrase-MiniLM-L6-v2')


def ensure_model():
    pass


def summarize_text(text, length):
    return model(text, num_sentences=length)
