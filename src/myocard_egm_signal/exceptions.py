"""Exceptions for signal conditions that no operator can work around.

These are raised, not signalled by a sentinel return value, because a
sentinel propagates. A ``+inf`` threshold flows into the next stage and
comes out the far end as a plausible-looking result — a zero-width
activation complex flagged as a *good* measurement — which then lands in
the duration distribution a study is trying to characterise. Failing at
the point of detection is the only version of this that stays visible.

Two distinct conditions, deliberately not one
---------------------------------------------
:class:`EmptySignalError` is a **programming error**. Nothing in the
pipeline legitimately produces a zero-length array; it means a bad slice
or an off-by-one in an index range upstream.

:class:`ConstantSignalError` is a **data condition**. A dead or
disconnected electrode records a flat line, and so does a channel clipped
to a rail. That is not a broken program — it is bad data, and it is
expected to occur when sweeping thousands of channels across a corpus. A
caller processing a record in bulk should catch this one specifically,
count it, and move to the next channel; a caller measuring a single trace
should let it surface.

Both derive from :class:`DegenerateSignalError` (itself a ``ValueError``)
so a caller can also be indifferent and catch the pair.

Why this diverges from the pooled-amplitude convention
------------------------------------------------------
The peak-to-peak threshold strategies in :mod:`.thresholds.healthy` and
:mod:`.thresholds.noise` return a sentinel for an empty *pool*. That is
kept, and the difference is intentional: an empty pool is a legitimate
outcome of filtering — no window passed — whereas an empty or flat
*signal* is not the outcome of anything. One is an answer, the other is
a defect.

Known limitation: whole-array only
----------------------------------
:class:`ConstantSignalError` fires only when the **entire** array is
constant. A trace that flatlines intermittently — an electrode dropping
out for a few seconds mid-record — passes this check, and every threshold
computed from it is then biased by the dead stretch without any warning.
Detecting bad *sections* needs a segment-wise analysis this library does
not yet do; see ``docs/theory.md`` §3.1.
"""

from __future__ import annotations


class DegenerateSignalError(ValueError):
    """A signal carries no information a threshold could be read from.

    Derives from ``ValueError`` so existing broad handlers still work.
    Catch this to be indifferent between the two causes; catch a
    subclass to distinguish a bug from bad data.
    """


class EmptySignalError(DegenerateSignalError):
    """A zero-length signal was passed. Almost certainly a caller bug."""


class ConstantSignalError(DegenerateSignalError):
    """Every sample holds the same value.

    A flat channel has no peaks, so there is nothing for a threshold to
    separate — no value would be meaningful. The usual physical causes
    are a dead or disconnected electrode (all zeros) and a channel
    saturated against a rail.
    """
