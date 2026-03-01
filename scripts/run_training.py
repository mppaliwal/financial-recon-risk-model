#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys

from recon_risk.app import run_training_from_csv
from recon_risk.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    input_csv = os.environ.get("RECON_INPUT_CSV", "").strip()
    if not input_csv:
        if len(sys.argv) < 2:
            raise ValueError(
                "Input CSV path required. Pass as argv[1] or set RECON_INPUT_CSV."
            )
        input_csv = sys.argv[1]
    print(json.dumps(run_training_from_csv(input_csv=input_csv), indent=2))


if __name__ == "__main__":
    main()
