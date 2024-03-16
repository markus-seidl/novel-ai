from textsum.summarize import Summarizer
from nltk import sent_tokenize
from tqdm.auto import tqdm, trange

SUMI: Summarizer = None


def ensure_model():
    global SUMI
    if SUMI is None:
        SUMI = Summarizer()

        print(f"Initialized textsum with: {SUMI.device}")


def summarize_text(text, length_chars) -> str:
    ensure_model()

    SUMI.set_inference_params({"max_length": length_chars})

    length_it = trange(3, leave=False)
    length_it.set_description("My Summaries ")
    for i in length_it:
        text = SUMI.summarize_string(text, disable_progress_bar=True)
        length_it.set_description("My Summaries (" + str(len(text)) + ")")
        if len(text) <= length_chars * 2 or length_chars < 0:
            return text

    return text


if __name__ == '__main__':
    # /Volumes/Dia/ai-data/anna-manual/
    # book_file = "aae8a9a0b14d2b900704cfc1e2ac3eb9.txt"
    # book_file = "../b0845a13375a4fb410e753ec526a8e3f.txt"
    # book_file = "055cc96d3c8a23505a6e6b353b773cd2.txt"
    # book_file = "a2a8b19cdddea509540191833a1364fc.txt"

    text = ("As I walked into the kitchen that morning, I had the strange feeling that something was going to happen. "
            "Maybe it had already happened and I just didn’t know it. I saw my wife standing there with a smirk on her "
            "beautiful face. When I saw her in the black negligee, I felt my downstairs begin to stir underneath my "
            " light"
            " cotton robe. \"Good Morning, darling. I’m excited to open my birthday gift.\" Still smiling, she "
            "seductively licked a drop of hot butter from her finger and said, \"It’d better be good.\" \"Uh-oh.\" "
            "I’d totally forgotten about it, and I realized she knew it. Stalling, I stole a slice of bacon from her "
            "plate, stuffed it in my mouth, mumbling incoherently around it.")

    summary = summarize_text(text, 200)
    print(summary)
