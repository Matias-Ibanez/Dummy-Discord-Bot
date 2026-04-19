import pytest
from app.utils import _take_two_sentences, _clean_leading


def test_take_two():
    s = "First sentence. Second sentence! Third?"
    assert _take_two_sentences(s) == "First sentence. Second sentence!"


def test_clean_leading():
    s = "Here you go: you're terrible."
    assert _clean_leading(s) == "you're terrible."
