"""
inspect_prompts.py
------------------
For each dataset, loads 3 distinct questions and shows ALL their true/false
propositions (i.e. every choice/answer option for that question).

Run from the Aligned_Comparison_BB_WB directory:
    python inspect_prompts.py
"""

import sys
from pathlib import Path
from collections import OrderedDict

# Force UTF-8 output so non-Latin characters in dataset text don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from load_data import load_dataset_aligned
from prompt_utils import get_white_box_context

DATASETS = [
    "commonsense_qa",
    "race",
    "arc_easy",
    "arc_challenge",
    "open_book_qa",
    "got_cities",
    "got_sp_en_trans",
    "got_larger_than",
    "imdb",
    "amazon_polarity",
    "ag_news",
    "dbpedia_14",
    "rte",
    "copa",
    "boolq",
]

N_QUESTIONS   = 3    # distinct questions (groups) to show
LOAD_BUDGET   = 300  # how many raw samples to load to guarantee 3 full groups
MAX_PROMPT_CHARS = 1200

SEPARATOR = "=" * 80
SUBSEP    = "-" * 60


def truncate(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated — {len(text) - limit} more chars]"


def get_first_n_groups(samples: list[dict], n: int) -> list[list[dict]]:
    """
    Groups samples by their 'id' (= group_id from the loader) and returns
    the first n groups in order of first appearance.
    """
    seen: OrderedDict[str, list[dict]] = OrderedDict()
    for row in samples:
        gid = str(row.get("id", "unknown"))
        seen.setdefault(gid, []).append(row)
        if len(seen) == n and all(len(v) > 0 for v in seen.values()):
            # Keep collecting until the current group is fully closed.
            # We detect closure when the next sample belongs to a new group
            # and we already have n groups — handled below.
            pass

    # Return only the first n groups
    groups = list(seen.values())[:n]
    return groups


def main():
    output_lines = []

    def emit(line: str = ""):
        print(line)
        output_lines.append(line)

    emit("PROMPT INSPECTION REPORT")
    emit(f"3 questions × all propositions per dataset  ({N_QUESTIONS * len(DATASETS)} questions total)")
    emit(SEPARATOR)

    for ds_name in DATASETS:
        emit()
        emit(SEPARATOR)
        emit(f"DATASET: {ds_name.upper()}")
        emit(SEPARATOR)

        samples = load_dataset_aligned(ds_name, split="train", n_samples=LOAD_BUDGET)

        if not samples:
            emit("  [ERROR] No samples loaded.")
            continue

        groups = get_first_n_groups(samples, N_QUESTIONS)

        for q_idx, group in enumerate(groups):
            emit()
            emit(f"  -- Question {q_idx + 1}  (group_id = {group[0].get('id', '?')}) --")

            for row in group:
                label_val = row.get("label", "?")
                label_str = "TRUE " if label_val == 1 else "FALSE"
                prompt = get_white_box_context(row)

                emit()
                emit(f"    [{label_str}]")
                emit(SUBSEP)
                # indent each line of the prompt for readability
                for line in truncate(prompt).splitlines():
                    emit("    " + line)

        emit()

    emit()
    emit(SEPARATOR)
    emit("END OF REPORT")
    emit(SEPARATOR)

    out_path = Path(__file__).parent / "prompt_inspection_report.txt"
    out_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
