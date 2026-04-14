"""Quantization tests."""

from kv_hierarchy_lab.quant.footprint import quantized_size_bytes


def test_quantized_size_bytes_orders_schemes() -> None:
    """Smaller quantized formats should consume fewer bytes."""
    fp16 = quantized_size_bytes(1024, "fp16")
    fp8 = quantized_size_bytes(1024, "fp8")
    int4 = quantized_size_bytes(1024, "int4")
    int2 = quantized_size_bytes(1024, "int2")
    assert fp16 > fp8 > int4 > int2
