from src.voice.stt.common.schemas import TranscriptResult


def test_final_transcript():
    result = TranscriptResult(
        text="hello",
        is_final=True,
        language="en",
    )

    assert result.text == "hello"
    assert result.is_final is True
    assert result.language == "en"
    assert result.type == "final"


def test_partial_transcript():
    result = TranscriptResult(
        text="hello",
        is_final=False,
        language="en",
    )

    assert result.type == "partial"


def test_to_dict():
    result = TranscriptResult(
        text="hello",
        is_final=True,
        language="en",
    )

    data = result.to_dict()

    assert data["type"] == "transcript"
    assert data["text"] == "hello"
    assert data["is_final"] is True
    assert data["language"] == "en"
