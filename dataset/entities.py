from dataclasses import dataclass
from typing import List


@dataclass
class Chapter:
    title: str
    sentences: List[str]
    summary: str = ""


@dataclass
class Book:
    title: str
    chapters: List[Chapter]


@dataclass
class TrainingData:
    chapter_title: str
    book_title: str
    summary: str
    previous_sentences: str
    expected_answer: str
