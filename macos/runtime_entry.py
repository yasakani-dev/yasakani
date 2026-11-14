# SPDX-License-Identifier: Apache-2.0
"""Entry point for the bundled CLI, independent of an installed Python."""
from yasakani.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
