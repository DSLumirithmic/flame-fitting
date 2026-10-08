"""Backport the Python 3.11 stdlib bits that gnm-shape uses, so it runs on 3.10.

This repo is pinned to Python 3.10 (chumpy needs inspect.getargspec, removed in
3.11), but the GNM shape library uses ``enum.StrEnum`` and ``typing.Self``
(both new in 3.11). Import this module before anything from ``gnm.shape``.
"""
import enum
import sys
import typing

if sys.version_info < (3, 11):
    if not hasattr(enum, 'StrEnum'):
        class StrEnum(str, enum.Enum):
            """Minimal backport of enum.StrEnum (3.11)."""

            def __new__(cls, *values):
                value = str(*values)
                member = str.__new__(cls, value)
                member._value_ = value
                return member

            def __str__(self):
                return str.__str__(self)

            @staticmethod
            def _generate_next_value_(name, start, count, last_values):
                return name.lower()

        enum.StrEnum = StrEnum

    if not hasattr(typing, 'Self'):
        from typing_extensions import Self
        typing.Self = Self
