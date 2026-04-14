"""Generate a synthetic trace and print simple summary stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_hierarchy_lab.workloads import (
    generate_adversarial_prefetch,
    generate_chat_continuation,
    generate_long_tail_mix,
    generate_periodic_reuse,
    generate_prefetch_friendly,
    generate_rag_burst,
)


WORKLOADS = {
    "chat_continuation": generate_chat_continuation,
    "rag_burst": generate_rag_burst,
    "periodic_reuse": generate_periodic_reuse,
    "long_tail_mix": generate_long_tail_mix,
    "prefetch_friendly": generate_prefetch_friendly,
    "adversarial_prefetch": generate_adversarial_prefetch,
}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, default="chat_continuation")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--quant", default="fp16")
    args = parser.parse_args()

    workload = WORKLOADS[args.workload](
        length=args.length,
        num_pages=args.num_pages,
        quantization_scheme=args.quant,
    )
    print(f"workload={workload.name} pages={len(workload.pages)} accesses={len(workload.accesses)}")
    print("first_10_accesses=", [access.page_id for access in workload.accesses[:10]])


if __name__ == "__main__":
    main()
