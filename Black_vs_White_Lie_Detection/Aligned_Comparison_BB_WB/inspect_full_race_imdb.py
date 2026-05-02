"""
inspect_full_race_imdb.py
-------------------------
Prints the full, untruncated prompts for 1 question from RACE and 1 from IMDB,
showing every true/false proposition for each question.

Run from the Aligned_Comparison_BB_WB directory:
    python inspect_full_race_imdb.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from load_data import load_dataset_aligned
from prompt_utils import get_white_box_context

SEPARATOR = "=" * 80
SUBSEP    = "-" * 60

lines = []

def emit(line: str = ""):
    print(line)
    lines.append(line)


def inspect_dataset(ds_name: str):
    emit()
    emit(SEPARATOR)
    emit(f"DATASET: {ds_name.upper()}  --  1 full question, all propositions (no truncation)")
    emit(SEPARATOR)

    samples = load_dataset_aligned(ds_name, split="train", n_samples=50)
    if not samples:
        emit("  [ERROR] No samples loaded.")
        return

    # Collect all rows that share the first group_id
    first_id = str(samples[0].get("id", "unknown"))
    group = [row for row in samples if str(row.get("id", "")) == first_id]

    emit(f"  group_id = {first_id}  |  {len(group)} proposition(s)")

    for row in group:
        label_val = row.get("label", "?")
        label_str = "TRUE " if label_val == 1 else "FALSE"
        prompt = get_white_box_context(row)

        emit()
        emit(f"  [{label_str}]")
        emit(SUBSEP)
        # Print every line of the prompt indented
        for pline in prompt.splitlines():
            emit("  " + pline)
        emit()


def main():
    emit("FULL PROMPT INSPECTION -- RACE & IMDB (no truncation)")
    emit(SEPARATOR)

    inspect_dataset("race")
    inspect_dataset("imdb")

    emit()
    emit(SEPARATOR)
    emit("END")
    emit(SEPARATOR)

    out_path = Path(__file__).parent / "inspect_full_race_imdb.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
