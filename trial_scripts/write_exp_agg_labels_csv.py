import argparse
import csv
import re
from pathlib import Path


def parse_expansion(value: str) -> int:
    try:
        expansion = int(float(value.strip()))
    except ValueError as exc:
        raise ValueError(f"Expansion label must be 0 or 1, got {value!r}") from exc

    if expansion not in (0, 1):
        raise ValueError(f"Expansion label must be 0 or 1, got {expansion}")
    return expansion


def aggregate_class_from_name(filename: str) -> int | None:
    stem = Path(filename).stem

    match = re.match(r"L([1-4])", stem)
    if not match:
        return None

    # User rule: L1 -> 0, L2 -> 1, L3 -> 2, L4 -> 2.
    return min(int(match.group(1)) - 1, 2)


def load_mapping(mapping_csv: Path | None) -> dict[str, str]:
    if mapping_csv is None:
        return {}
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    mapping = {}
    with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"index", "original_stem"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV is missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            index = row["index"].strip()
            original_stem = row["original_stem"].strip()
            if not index or not original_stem:
                continue

            mapping[f"{index}.png"] = original_stem

    return mapping


def row_values(row: list[str], row_number: int) -> tuple[str, int]:
    if len(row) < 2:
        raise ValueError(f"Input CSV row {row_number} must have at least 2 columns: filename, expansion")

    filename = row[0].strip()
    if not filename:
        raise ValueError(f"Input CSV row {row_number} has an empty filename")

    return filename, parse_expansion(row[1])


def write_labels(input_csv: Path, output_csv: Path, mapping_csv: Path | None) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    final_to_original = load_mapping(mapping_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_csv.open("r", encoding="utf-8", newline="") as input_handle, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.reader(input_handle)
        writer = csv.writer(output_handle)
        writer.writerow(["filename", "expansion", "aggregate_class", "combo_id"])

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                filename, expansion = row_values(row, row_number)
            except ValueError:
                if row_number == 1:
                    continue
                raise

            aggregate_source = final_to_original.get(filename, filename)
            aggregate_class = aggregate_class_from_name(aggregate_source)
            if aggregate_class is None:
                raise ValueError(
                    f"Could not infer aggregate class for {filename!r} from {aggregate_source!r}. "
                    "Expected original names starting with L1, L2, L3, or L4. "
                    "For numbered filenames, pass --mapping-csv."
                )

            combo_id = aggregate_class * 2 + expansion
            writer.writerow([filename, expansion, aggregate_class, combo_id])
            rows_written += 1

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write expansion, aggregate class, and combo labels from a CSV whose first two columns are "
            "final filename and expansion label."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/label.csv"),
        help="Input CSV. Column 1 is final filename, column 2 is expansion label.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/metadata_exp_agg_combo.csv"),
        help="Output CSV with filename, expansion, aggregate_class, combo_id.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("/home/jixi/project/genai/trial_scripts/filename_mapping.csv"),
        help="Optional CSV mapping final numbered names to original_stem names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = write_labels(args.input_csv, args.output_csv, args.mapping_csv)
    print(f"Wrote {rows_written} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
