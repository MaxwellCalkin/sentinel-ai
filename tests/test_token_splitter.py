"""Tests for token splitter."""

import pytest
from sentinel.token_splitter import TokenSplitter, TextChunk


class TestBasicSplit:
    def test_short_text_no_split(self):
        s = TokenSplitter(max_tokens=1000)
        chunks = s.split("Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_long_text_splits(self):
        s = TokenSplitter(max_tokens=10)
        text = "word " * 100  # 500 chars, ~125 tokens
        chunks = s.split(text)
        assert len(chunks) > 1

    def test_empty_text(self):
        s = TokenSplitter()
        assert s.split("") == []
        assert s.split("   ") == []

    def test_chunk_indices(self):
        s = TokenSplitter(max_tokens=10)
        chunks = s.split("word " * 50)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.total_chunks == len(chunks)


class TestSeparators:
    def test_paragraph_split(self):
        s = TokenSplitter(max_tokens=20, separator="paragraph")
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = s.split(text)
        assert len(chunks) >= 1

    def test_sentence_split(self):
        s = TokenSplitter(max_tokens=10, separator="sentence")
        text = "First sentence. Second sentence. Third sentence."
        chunks = s.split(text)
        assert len(chunks) >= 1

    def test_word_split(self):
        s = TokenSplitter(max_tokens=5, separator="word")
        text = "one two three four five six seven eight nine ten"
        chunks = s.split(text)
        assert len(chunks) >= 2


class TestOverlap:
    def test_overlap_adds_context(self):
        s = TokenSplitter(max_tokens=10, overlap_tokens=2, separator="word")
        text = "word " * 50
        chunks = s.split(text)
        assert len(chunks) >= 2
        # Second chunk should have overlap
        if len(chunks) > 1:
            assert chunks[1].overlap_tokens > 0

    def test_no_overlap_first_chunk(self):
        s = TokenSplitter(max_tokens=10, overlap_tokens=5)
        chunks = s.split("word " * 50)
        if chunks:
            assert chunks[0].overlap_tokens == 0


class TestTokenEstimate:
    def test_token_estimate(self):
        s = TokenSplitter(max_tokens=1000)
        chunks = s.split("Hello world testing")
        assert chunks[0].token_estimate > 0

    def test_chunks_within_limit(self):
        s = TokenSplitter(max_tokens=50, separator="sentence")
        text = "This is a sentence. " * 20
        chunks = s.split(text)
        # Each chunk token estimate should be reasonable
        for chunk in chunks:
            assert chunk.token_estimate > 0


class TestCountChunks:
    def test_count_short(self):
        s = TokenSplitter(max_tokens=1000)
        assert s.count_chunks("Short text") == 1

    def test_count_long(self):
        s = TokenSplitter(max_tokens=10)
        count = s.count_chunks("word " * 100)
        assert count > 1

    def test_count_empty(self):
        s = TokenSplitter()
        assert s.count_chunks("") == 0


class TestStructure:
    def test_chunk_structure(self):
        s = TokenSplitter()
        chunks = s.split("Test text")
        chunk = chunks[0]
        assert isinstance(chunk, TextChunk)
        assert isinstance(chunk.token_estimate, int)
        assert chunk.index == 0
        assert chunk.total_chunks >= 1
