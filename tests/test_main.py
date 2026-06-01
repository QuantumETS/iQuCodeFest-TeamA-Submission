"""This module contains tests for the main module."""

import pytest

from main import sum


class TestSum:
    """Tests for the sum function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (1, 2, 3),
            (0, 0, 0),
            (-1, -1, -2),
            (-1, 1, 0),
        ],
    )
    def test_sum(self, a, b, expected):
        """Test the sum function."""
        assert sum(a, b) == expected
