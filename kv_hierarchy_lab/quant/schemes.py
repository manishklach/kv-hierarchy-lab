"""Quantization scheme definitions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantizationScheme:
    """Defines simple footprint and decode-overhead properties."""

    name: str
    bytes_per_scalar: float
    decode_penalty_ms: float = 0.0


QUANTIZATION_SCHEMES: dict[str, QuantizationScheme] = {
    "fp16": QuantizationScheme(name="fp16", bytes_per_scalar=2.0, decode_penalty_ms=0.00),
    "fp8": QuantizationScheme(name="fp8", bytes_per_scalar=1.0, decode_penalty_ms=0.01),
    "int4": QuantizationScheme(name="int4", bytes_per_scalar=0.5, decode_penalty_ms=0.03),
    "int2": QuantizationScheme(name="int2", bytes_per_scalar=0.25, decode_penalty_ms=0.06),
}
