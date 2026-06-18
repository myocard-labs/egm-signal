"""Record Protocol — the structural type the extractors consume.

The healthy- and noise-segment extractors operate on "a record" — some
object that exposes a multi-channel signal, knows its own sample rate,
and can map channel names to column indices. Concrete record types
(``myocard_iafdb_pipeline.records.IAFDBRecord`` today, future
producers tomorrow) satisfy this Protocol structurally — no explicit
``isinstance`` registration needed.

The Protocol is intentionally minimal: only the fields the extractors
read. Producers can carry any other state they want (drug-delivery
comments, ADC gains, QRS annotations) and the extractors will ignore
those fields.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Record(Protocol):
    """A multi-channel signal record consumable by the extractors.

    Attributes are declared as read-only ``@property``s rather than bare
    ``var: T`` so frozen-dataclass implementations (e.g.
    ``IAFDBRecord``) satisfy the Protocol cleanly. Mutable dataclasses
    work too — the read-only Protocol matches any narrower
    implementation.

    Attributes
    ----------
    name
        Identifier carried through into every produced segment so
        downstream consumers can trace a segment back to its source
        (e.g. ``"iaf1_afw"``).
    patient
        Patient-level grouping key, used by downstream patient-aware
        splits (e.g. ``"iaf1"``).
    fs
        Sampling frequency in Hz.
    signal
        Array of shape ``(n_samples, n_channels)``. Producers decide
        whether this is in calibrated mV or raw signal units; the
        extractors don't assume either. Calibration (if applied) is the
        caller's responsibility via a separate :class:`Calibration`.
    channel_names
        Channel labels parallel to the columns of :attr:`signal`. A
        tuple is recommended for hashability + immutability but any
        sequence is acceptable.
    """

    @property
    def name(self) -> str: ...
    @property
    def patient(self) -> str: ...
    @property
    def fs(self) -> float: ...
    @property
    def signal(self) -> np.ndarray: ...
    @property
    def channel_names(self) -> tuple[str, ...]: ...

    def channel_index(self, name: str) -> int:
        """Return the column index of ``name`` in :attr:`signal`.

        Implementations may raise ``KeyError`` (or any subclass) if the
        channel is not present. The extractors filter against
        :attr:`channel_names` before calling this so the lookup should
        not fail in practice.
        """
        ...
