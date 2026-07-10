"""Producer-contract types -- re-exported from reishi.execution.contract, the
canonical home now that a Producer is understood to be a pure function of
the manifest; oyster's own heartbeat/reap supervision is layered on by its
worker, not part of this contract type itself.
"""

from reishi.execution.contract import Producer, ProducerResult

__all__ = ["Producer", "ProducerResult"]
