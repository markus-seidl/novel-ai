from summarizer.sbert import SBertSummarizer

model = SBertSummarizer('paraphrase-MiniLM-L6-v2')


def summarize_text(text, length):
    return model(text, num_sentences=length)
