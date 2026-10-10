# tests/test_input_guard.py
import pytest

from input_guard import check_allowed_chars

INPUT_MAX_LENGTH = 1024


def test_1025_chars_should_refuse():
    """
    輸入 1025 個字，應該因為超過 1024 字而被拒絕。
    """
    text = "測" * 1025

    assert len(text) == 1025

    result = check_allowed_chars(
        text,
        allow_empty=False,
        max_length=INPUT_MAX_LENGTH,
    )

    assert result == "refuse"


def test_1024_chars_should_accept():
    """
    輸入 1024 個字，應該通過文長限制。
    """
    text = "測" * 1024

    assert len(text) == 1024

    result = check_allowed_chars(
        text,
        allow_empty=False,
        max_length=INPUT_MAX_LENGTH,
    )

    assert result == "accept"


@pytest.mark.parametrize(
    "length, expected",
    [
        (0, "refuse"),
        (1, "accept"),
        (100, "accept"),
        (1024, "accept"),
        (1025, "refuse"),
        (2000, "refuse"),
    ],
)
def test_length_boundary(length, expected):
    """
    測試不同長度邊界。
    """
    text = "A" * length

    result = check_allowed_chars(
        text,
        allow_empty=False,
        max_length=INPUT_MAX_LENGTH,
    )

    assert result == expected
