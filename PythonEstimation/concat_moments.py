"""concat_moments.py

Concatenate chunk CSV files produced by the simulate-moments job array.
Keeps the header from the first file and appends data rows from all chunks.
Python translation of concat_moments.jl.
"""

import glob
import os
import sys

INPUT_DIR = "SimulatedData"
OUTPUT_PATH = os.path.join(INPUT_DIR, "moments.csv")


def main():
    chunk_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "moments_*.csv"))
    )

    if not chunk_files:
        sys.exit(f"No chunk files found in {INPUT_DIR}/")

    header_line = None
    rows_written = 0

    with open(OUTPUT_PATH, "w") as fout:
        for fpath in chunk_files:
            with open(fpath, "r") as fin:
                first = True
                for line in fin:
                    if first:
                        first = False
                        if header_line is None:
                            header_line = line
                            fout.write(line)
                        elif line != header_line:
                            sys.exit(f"Header mismatch in {fpath}")
                        continue
                    fout.write(line)
                    rows_written += 1

    print(
        f"Concatenated {len(chunk_files)} files -> {OUTPUT_PATH} "
        f"({rows_written} rows)"
    )


if __name__ == "__main__":
    main()
