"""Comprehensive focused unit tests for StreamingPhraseBuffer (Phase 5C - Phase 1).

Verifies:
1. Multiple tokens accumulating to form single words across fragment boundaries.
2. Sentence-ending punctuation ('.', '!', '?', and '\\n') triggering prompt flushes.
3. Clause punctuation (',', ';', ':') enforcing minimum word count (>= 3 words).
4. Run-on word ceiling (default 8 words) flushing at complete word boundaries.
5. End-of-stream flush emitting trailing text without losing characters.
6. Whitespace normalization, empty fragments, and no empty/whitespace-only chunks.
7. Decimal numbers ('1.5', '3.14') and time expressions ('10:30') preserved without splitting.
8. Common abbreviations ('Mr.', 'Dr.', 'sq.', 'yd.', 'cr.', 'e.g.') avoiding false sentence breaks.
9. Preservation of text order, punctuation, and character integrity (no lost or duplicate text).
10. Multiple sentences arriving in a single fragment.
11. Very long text streams without punctuation.
12. Reset and state cleanliness.
"""

from __future__ import annotations

import pytest

from src.voice.streaming_buffer import StreamingPhraseBuffer


# =============================================================================
# 1. Fragment Accumulation & Word Completion
# =============================================================================

def test_multiple_tokens_forming_single_words():
    """Tokens arriving in tiny sub-word slices accumulate into whole words before flushing."""
    buffer = StreamingPhraseBuffer()

    assert buffer.feed("Hy") == []
    assert buffer.feed("der") == []
    assert buffer.feed("abad") == []
    assert buffer.feed(" is") == []
    assert buffer.feed(" a") == []
    assert buffer.feed(" city.") == ["Hyderabad is a city."]
    assert buffer.is_empty


def test_split_word_across_fragments_at_sentence_boundary():
    """Word completion at punctuation boundary does not drop or repeat characters."""
    buffer = StreamingPhraseBuffer()

    r1 = buffer.feed("We have lux")
    assert r1 == []
    r2 = buffer.feed("ury villas.")
    assert r2 == ["We have luxury villas."]
    assert buffer.flush() == []


# =============================================================================
# 2. Sentence-Ending Punctuation & Newlines
# =============================================================================

@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Hello world.", "Hello world."),
        ("Is this HMDA approved?", "Is this HMDA approved?"),
        ("Yes, absolutely!", "Yes, absolutely!"),
        ("First sentence.\nSecond sentence.", ["First sentence.", "Second sentence."]),
    ],
)
def test_sentence_ending_punctuation(sentence, expected):
    """Period, question mark, exclamation mark, and newlines trigger sentence flushes."""
    buffer = StreamingPhraseBuffer()
    res = buffer.feed(sentence)
    res.extend(buffer.flush())
    if isinstance(expected, list):
        assert res == expected
    else:
        assert res == [expected]


def test_multiple_sentences_in_single_fragment():
    """A single fragment containing multiple sentences splits into distinct phrase chunks."""
    buffer = StreamingPhraseBuffer()
    fragment = "Welcome to Infra Builder. We offer luxury plots! Would you like a brochure?"
    chunks = buffer.feed(fragment)

    assert chunks == [
        "Welcome to Infra Builder.",
        "We offer luxury plots!",
        "Would you like a brochure?",
    ]
    assert buffer.is_empty


def test_closing_quotes_and_brackets_attached_to_sentence():
    """Quotes, parens, and brackets following punctuation attach cleanly to the chunk."""
    buffer = StreamingPhraseBuffer()
    res = buffer.feed('He said, "Welcome to Pristine City!" (It is near TSPA.) ')
    assert res == [
        'He said, "Welcome to Pristine City!"',
        "(It is near TSPA.)",
    ]


# =============================================================================
# 3. Clause Punctuation & Minimum Word Count
# =============================================================================

def test_clause_punctuation_below_minimum_words_does_not_flush():
    """Commas or semicolons with fewer than min_clause_words remain in buffer until sentence end."""
    buffer = StreamingPhraseBuffer(min_clause_words=3)

    # 1 word before comma -> hold
    assert buffer.feed("Yes, ") == []
    assert buffer.feed("we do.") == ["Yes, we do."]

    # 2 words before comma -> hold
    buffer.reset()
    assert buffer.feed("Good morning, ") == []
    assert buffer.feed("how can I help?") == ["Good morning, how can I help?"]


def test_clause_punctuation_with_sufficient_words_flushes():
    """Commas or semicolons with at least min_clause_words trigger an incremental clause flush."""
    buffer = StreamingPhraseBuffer(min_clause_words=3)

    # 5 words before comma -> flush clause
    r1 = buffer.feed("In addition to Pristine City, ")
    assert r1 == ["In addition to Pristine City,"]

    # Remainder flushed at period
    r2 = buffer.feed("we also offer Serene City.")
    assert r2 == ["we also offer Serene City."]
    assert buffer.is_empty


def test_semicolon_and_colon_clause_flushing():
    """Semicolons and colons act as clause boundaries when word threshold is met."""
    buffer = StreamingPhraseBuffer(min_clause_words=3)

    res = buffer.feed("We have three projects; each layout is HMDA approved: JB Pride is luxury.")
    assert res == [
        "We have three projects;",
        "each layout is HMDA approved:",
        "JB Pride is luxury.",
    ]


# =============================================================================
# 4. Word Count Ceiling
# =============================================================================

def test_eight_word_ceiling_flushes_at_word_boundary():
    """Run-on text without punctuation flushes in 8-word chunks at clean word boundaries."""
    buffer = StreamingPhraseBuffer(max_words_ceiling=8)
    words = [f"word{i}" for i in range(1, 19)]  # 18 words total

    chunks = []
    for w in words:
        chunks.extend(buffer.feed(w + " "))
    chunks.extend(buffer.flush())

    assert len(chunks) == 3
    assert chunks[0] == "word1 word2 word3 word4 word5 word6 word7 word8"
    assert chunks[1] == "word9 word10 word11 word12 word13 word14 word15 word16"
    assert chunks[2] == "word17 word18"


def test_very_long_text_without_punctuation():
    """A stream of 25+ words with zero punctuation emits sequential 8-word chunks."""
    buffer = StreamingPhraseBuffer(max_words_ceiling=8)
    words = [f"item{i}" for i in range(1, 26)]  # 25 words: 8 + 8 + 8 + 1

    chunks = buffer.feed(" ".join(words)) + buffer.flush()
    assert [len(c.split()) for c in chunks] == [8, 8, 8, 1]
    assert " ".join(chunks) == " ".join(words)


def test_eight_word_ceiling_does_not_chop_partial_words():
    """An 8th word that is still arriving without trailing space does not flush prematurely."""
    buffer = StreamingPhraseBuffer(max_words_ceiling=8)

    # 7 words + partial 8th word
    r1 = buffer.feed("one two three four five six seven eight")
    assert r1 == []  # 'eight' has no space or trailing word yet; might be 'eighteen'

    # 8th word completed by incoming text
    r2 = buffer.feed("een is a number.")
    # Word 8 is 'eighteen', followed by space, so ceiling triggers
    assert r2 == ["one two three four five six seven eighteen", "is a number."]


def test_punctuation_takes_precedence_over_word_ceiling():
    """If punctuation occurs at or before the word ceiling, it flushes at punctuation instead."""
    buffer = StreamingPhraseBuffer(min_clause_words=3, max_words_ceiling=8)

    # Clause punctuation at word 4 (< 8)
    res = buffer.feed("We offer residential plots, luxury villas, and apartments.")
    assert res == [
        "We offer residential plots,",
        "luxury villas, and apartments.",
    ]


# =============================================================================
# 5. Decimal Numbers, Time Expressions & Numbers with Commas
# =============================================================================

def test_decimal_numbers_not_split():
    """Periods inside decimal numbers (e.g. 1.5, 3.14) do not trigger false sentence breaks."""
    # Test within default 8-word ceiling
    buffer = StreamingPhraseBuffer()
    res = buffer.feed("The plot covers 1.5 acres.")
    assert res == ["The plot covers 1.5 acres."]

    # Test multi-decimal sentence with expanded ceiling
    buffer_wide = StreamingPhraseBuffer(max_words_ceiling=15)
    res_wide = buffer_wide.feed("The plot is 1.5 acres and costs 2.4 crore.")
    assert res_wide == ["The plot is 1.5 acres and costs 2.4 crore."]


def test_decimal_split_across_token_boundary():
    """A decimal number whose dot arrives at the end of a fragment is not split."""
    buffer = StreamingPhraseBuffer()

    r1 = buffer.feed("The price is 1.")
    assert r1 == []  # Dot after digit at end of buffer: must wait
    r2 = buffer.feed("5 crore per unit.")
    assert r2 == ["The price is 1.5 crore per unit."]


def test_time_format_and_numbers_with_commas_preserved():
    """Colons in time ('10:30') and commas in numbers ('10,000') do not trigger clause flushes."""
    # Test within default ceiling
    buffer = StreamingPhraseBuffer(min_clause_words=3)
    res = buffer.feed("Please meet at 10:30 AM with 50,000 rupees.")
    assert res == ["Please meet at 10:30 AM with 50,000 rupees."]

    # Test longer sentence with expanded ceiling
    buffer_wide = StreamingPhraseBuffer(min_clause_words=3, max_words_ceiling=15)
    res_wide = buffer_wide.feed("Please meet at 10:30 AM with a deposit of 50,000 rupees.")
    assert res_wide == ["Please meet at 10:30 AM with a deposit of 50,000 rupees."]


def test_time_colon_split_across_token_boundary():
    """Colon in time format arriving at fragment boundary waits for trailing digits."""
    buffer = StreamingPhraseBuffer(min_clause_words=3)

    r1 = buffer.feed("Site visit at 10:")
    assert r1 == []  # Colon after digit at end of buffer: wait
    r2 = buffer.feed("30 AM today.")
    assert r2 == ["Site visit at 10:30 AM today."]


# =============================================================================
# 6. Abbreviations and Titles
# =============================================================================

def test_titles_do_not_trigger_sentence_breaks():
    """Titles like Mr., Dr., Prof. do not break sentences even when followed by capitalized names."""
    buffer = StreamingPhraseBuffer()

    res = buffer.feed("Dr. Rao and Mr. Sharma visited today.")
    assert res == ["Dr. Rao and Mr. Sharma visited today."]


def test_common_unit_abbreviations():
    """Units like sq. yards, ft., and cr. do not trigger false sentence breaks."""
    # Test within default ceiling
    buffer = StreamingPhraseBuffer()
    res = buffer.feed("The villa is 2500 sq. ft. in area.")
    assert res == ["The villa is 2500 sq. ft. in area."]

    # Test multi-unit sentence with expanded ceiling
    buffer_wide = StreamingPhraseBuffer(max_words_ceiling=20)
    res_wide = buffer_wide.feed("The villa is 2500 sq. ft. on a 300 sq. yd. plot costing 2.5 cr. total.")
    assert res_wide == ["The villa is 2500 sq. ft. on a 300 sq. yd. plot costing 2.5 cr. total."]


def test_abbreviation_at_sentence_boundary_followed_by_capital():
    """An abbreviation followed by whitespace and a capital letter allows a valid sentence break."""
    buffer = StreamingPhraseBuffer()

    res = buffer.feed("The villa costs 1.5 cr. Each unit has premium amenities.")
    assert res == [
        "The villa costs 1.5 cr.",
        "Each unit has premium amenities.",
    ]


def test_single_letter_initials_preserved():
    """Single-letter initials like 'J.B.' do not trigger false sentence breaks."""
    buffer = StreamingPhraseBuffer()

    res = buffer.feed("J.B. Infra is the premier builder in Hyderabad.")
    assert res == ["J.B. Infra is the premier builder in Hyderabad."]


# =============================================================================
# 7. End-of-Stream Flush & State Cleanliness
# =============================================================================

def test_end_of_stream_flush_emits_remaining_text():
    """flush() outputs all remaining valid text and resets the buffer."""
    buffer = StreamingPhraseBuffer()

    assert buffer.feed("Thank you for contacting Infra Builder") == []
    assert not buffer.is_empty

    flushed = buffer.flush()
    assert flushed == ["Thank you for contacting Infra Builder"]
    assert buffer.is_empty
    assert buffer.flush() == []


def test_whitespace_normalization_and_empty_fragments():
    """Empty fragments, whitespace-only inputs, and extraneous spacing never emit blank chunks."""
    buffer = StreamingPhraseBuffer()

    assert buffer.feed("") == []
    assert buffer.feed("   ") == []
    assert buffer.feed("\t  \t") == []

    res = buffer.feed("Hello world!   ")
    assert res == ["Hello world!"]
    assert buffer.flush() == []


def test_reset_clears_buffer():
    """reset() wipes internal state without emitting chunks."""
    buffer = StreamingPhraseBuffer()
    buffer.feed("Unfinished sentence that should be")
    assert not buffer.is_empty

    buffer.reset()
    assert buffer.is_empty
    assert buffer.flush() == []


# =============================================================================
# 8. Integrity: No Lost or Duplicated Text
# =============================================================================

def test_character_and_word_integrity_across_arbitrary_chunking():
    """Arbitrary character-level chunking preserves complete passage without data loss or duplication."""
    buffer = StreamingPhraseBuffer(min_clause_words=3, max_words_ceiling=8)
    passage = (
        "Infra Builder is a trusted developer in Hyderabad. "
        "We offer luxury villas, gated community plots, and commercial complexes with world-class amenities! "
        "Please call us at 10:30 AM for a site visit."
    )

    # Slice into irregular fragment lengths
    import random
    random.seed(42)

    pos = 0
    fragments = []
    while pos < len(passage):
        step = random.randint(1, 6)
        fragments.append(passage[pos : pos + step])
        pos += step

    emitted_chunks = []
    for frag in fragments:
        emitted_chunks.extend(buffer.feed(frag))
    emitted_chunks.extend(buffer.flush())

    reconstructed_words = " ".join(emitted_chunks).split()
    original_words = passage.split()

    assert reconstructed_words == original_words


# =============================================================================
# 9. Configuration Validation
# =============================================================================

def test_invalid_configuration_raises_value_error():
    """Invalid min/max configuration raises ValueError at initialization."""
    with pytest.raises(ValueError, match="min_clause_words must be at least 1"):
        StreamingPhraseBuffer(min_clause_words=0)

    with pytest.raises(ValueError, match="max_words_ceiling must be greater"):
        StreamingPhraseBuffer(min_clause_words=5, max_words_ceiling=3)
