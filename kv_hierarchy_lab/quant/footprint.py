"""Footprint calculations for quantized KV pages."""

from kv_hierarchy_lab.quant.schemes import QUANTIZATION_SCHEMES


def quantized_size_bytes(num_scalars: int, quantization_scheme: str) -> int:
    """Returns the modeled page footprint in bytes."""
    scheme = QUANTIZATION_SCHEMES[quantization_scheme]
    return max(1, int(num_scalars * scheme.bytes_per_scalar))
