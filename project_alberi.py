"""
project_alberi.py
=================

Data Analysis Project (Intro to Python Programming)
Dataset: Monumental Trees of Sicily

Constraints (as required by the assignment)
-------------------------------------------
- Do NOT use pandas
- Do NOT use the csv module
- You must use scikit-learn (sklearn) for KMeans
- You must model:
  - a single dataset record with classes (including geographic inheritance)
  - bounding boxes using classes (with inheritance/composition)

Features implemented
--------------------
1) Read the dataset from a CSV file WITHOUT using the csv module.
2) Convert dataset rows into a list of objects (N objects, one per tree).
3) Print number of trees available.
4) Bar charts: group by
   - Province
   - Urban context
   - Public interest proposal field
   User can save plots and must choose a valid file format.
5) Interval frequency (histogram-style with bar chart) for:
   - Circumference (cm)
   - Height (m)
   User provides N intervals, validated in [5..10].
   The plot can be saved in a validated image format.
6) Compute and print min, max, mean, std for circumference and height.
7) For k in [2..7]:
   - Run KMeans on circumference
   - Print cluster sizes
   - Write a CSV-text file (no csv module) with:
     K, cluster_id, nElements, centroidCircumference
8) Ask user position (lat/lon) with validation and a species name chosen from
   dataset values ("SPECIE NOME VOLGARE"). If invalid, ask again.
   Then print trees of that species and distance (Haversine). User can sort by:
   - distance ascending/descending
   - circumference ascending/descending
9) Read bounding boxes from a user-specified file.
   - Invalid bounding boxes are ignored
   - For each valid box: count trees inside and compute stats (height + circumference)
   - Save results to a JSON file name specified by user

The code is organized into small, reusable functions and includes main().

How to run
----------
pip install scikit-learn matplotlib
python project_alberi.py
"""

import math
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# =========================
#   Exceptions / Validation
# =========================

class ValidationError(ValueError):
    """Custom exception for invalid input or invalid data conditions."""
    pass


def validate_lat(lat: float) -> None:
    """Validate latitude range."""
    if not isinstance(lat, (int, float)) or math.isnan(lat) or lat < -90 or lat > 90:
        raise ValidationError("Invalid latitude (must be between -90 and 90).")


def validate_lon(lon: float) -> None:
    """Validate longitude range."""
    if not isinstance(lon, (int, float)) or math.isnan(lon) or lon < -180 or lon > 180:
        raise ValidationError("Invalid longitude (must be between -180 and 180).")


def validate_bins(n: int) -> None:
    """Validate number of intervals N (must be integer in [5..10])."""
    if not isinstance(n, int):
        raise ValidationError("N must be an integer.")
    if n < 5 or n > 10:
        raise ValidationError("N must be between 5 and 10 (inclusive).")


def validate_image_format(fmt: str) -> str:
    """
    Validate user-selected image format for saving plots.
    Returns a normalized format string.
    """
    fmt = (fmt or "").strip().lower().lstrip(".")
    allowed = {"png", "jpg", "jpeg", "pdf", "svg"}
    if fmt not in allowed:
        raise ValidationError(f"Invalid format. Choose one of: {', '.join(sorted(allowed))}")
    # Normalize jpeg to jpg
    if fmt == "jpeg":
        fmt = "jpg"
    return fmt


def validate_output_extension(filename: str, ext: str) -> None:
    """Ensure filename ends with the required extension (e.g., .csv or .json)."""
    if not filename.lower().endswith(ext.lower()):
        raise ValidationError(f"Output file must end with {ext}")


# =================================
#   CSV Parser (NO csv module)
# =================================

def parse_csv_text(text: str, delimiter: str = ",", quote: str = '"') -> List[List[str]]:
    """
    Minimal CSV parser (no csv module).
    Supports:
      - delimiter separation
      - quoted fields
      - escaped quotes via "" inside quoted fields
      - newline inside quoted fields

    Returns:
      rows: list of rows, each row is a list[str]
    """
    rows: List[List[str]] = []
    current_row: List[str] = []
    current_field_chars: List[str] = []
    in_quotes = False

    i = 0
    while i < len(text):
        ch = text[i]

        if ch == quote:
            # Handle escaped quote when inside quotes and next is quote
            if in_quotes and i + 1 < len(text) and text[i + 1] == quote:
                current_field_chars.append(quote)
                i += 1
            else:
                in_quotes = not in_quotes

        elif ch == delimiter and not in_quotes:
            # Field ends
            current_row.append("".join(current_field_chars).strip())
            current_field_chars = []

        elif ch == "\n" and not in_quotes:
            # Row ends
            current_row.append("".join(current_field_chars).strip())
            current_field_chars = []

            # Ignore fully empty lines
            if not (len(current_row) == 1 and current_row[0] == ""):
                rows.append(current_row)
            current_row = []

        else:
            current_field_chars.append(ch)

        i += 1

    # Add the last field/row if needed
    if current_field_chars or current_row:
        current_row.append("".join(current_field_chars).strip())
        if not (len(current_row) == 1 and current_row[0] == ""):
            rows.append(current_row)

    return rows


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read a text file safely (replace errors to avoid crashes on odd characters)."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


# =========================
#   Domain model (Classes)
# =========================

@dataclass
class GeoEntity:
    """
    Base class for objects that have geographic coordinates.
    This class is used to demonstrate inheritance for geographic data.
    """
    _lat: float
    _lon: float

    def __post_init__(self):
        validate_lat(self._lat)
        validate_lon(self._lon)

    # Getters
    def get_lat(self) -> float:
        return self._lat

    def get_lon(self) -> float:
        return self._lon

    # Setters
    def set_lat(self, lat: float) -> None:
        validate_lat(lat)
        self._lat = float(lat)

    def set_lon(self, lon: float) -> None:
        validate_lon(lon)
        self._lon = float(lon)


@dataclass
class MonumentalTree(GeoEntity):
    """
    Represents one monumental tree record from the dataset.
    Inherits from GeoEntity to reuse geographic logic (inheritance requirement).
    circumference and height are stored as float (assignment requirement).
    """
    _province: str
    _urban_context: str
    _species_common_name: str
    _public_interest_proposal: str
    _circumference_cm: float
    _height_m: float
    _town: str = ""
    _locality: str = ""

    # Getters / setters (explicitly included as requested)
    def get_province(self) -> str:
        return self._province

    def set_province(self, v: str) -> None:
        self._province = (v or "").strip()

    def get_urban_context(self) -> str:
        return self._urban_context

    def set_urban_context(self, v: str) -> None:
        self._urban_context = (v or "").strip()

    def get_species_common_name(self) -> str:
        return self._species_common_name

    def set_species_common_name(self, v: str) -> None:
        self._species_common_name = (v or "").strip()

    def get_public_interest_proposal(self) -> str:
        return self._public_interest_proposal

    def set_public_interest_proposal(self, v: str) -> None:
        self._public_interest_proposal = (v or "").strip()

    def get_circumference(self) -> float:
        return self._circumference_cm

    def set_circumference(self, v: float) -> None:
        self._circumference_cm = float(v)

    def get_height(self) -> float:
        return self._height_m

    def set_height(self, v: float) -> None:
        self._height_m = float(v)

    def get_town(self) -> str:
        return self._town

    def get_locality(self) -> str:
        return self._locality

    def short_str(self) -> str:
        """Short printable representation for lists and distance output."""
        return (
            f"{self._species_common_name} | Province: {self._province} | "
            f"Town: {self._town} | Circ: {self._circumference_cm:.1f} cm | "
            f"Height: {self._height_m:.1f} m"
        )


@dataclass
class GeoPoint:
    """
    A simple geographic point used inside BoundingBox (composition requirement).
    Note: uses lon, lat ordering explicitly.
    """
    lon: float
    lat: float

    def __post_init__(self):
        validate_lon(self.lon)
        validate_lat(self.lat)


@dataclass
class BoundingBox:
    """
    Bounding box defined by two corners:
      - top_left (lon, lat)
      - bottom_right (lon, lat)

    Composition: uses GeoPoint objects for corners.
    """
    top_left: GeoPoint
    bottom_right: GeoPoint

    def is_valid_box(self) -> bool:
        """
        A valid box must satisfy:
          top_left.lon < bottom_right.lon  (left is smaller)
          top_left.lat > bottom_right.lat  (top is larger latitude)
        """
        return (self.top_left.lon < self.bottom_right.lon) and (self.top_left.lat > self.bottom_right.lat)

    def contains(self, lon: float, lat: float) -> bool:
        """
        Check if a point is inside the box (inclusive boundaries).
        """
        return (
            self.top_left.lon <= lon <= self.bottom_right.lon and
            self.bottom_right.lat <= lat <= self.top_left.lat
        )


# =========================
#   Generic analytics
# =========================

def group_counts(items: List[MonumentalTree], key_fn) -> Dict[str, int]:
    """
    Count occurrences by a key extracted with key_fn.
    Uses a dictionary for counting.
    """
    counts: Dict[str, int] = {}
    for it in items:
        k = (key_fn(it) or "").strip()
        if k == "":
            k = "NOT SPECIFIED"
        counts[k] = counts.get(k, 0) + 1
    return counts


def bar_chart_from_counts(counts: Dict[str, int], title: str, xlabel: str, ylabel: str = "Count") -> None:
    """
    Display a bar chart using matplotlib from a dictionary of counts.
    """
    keys = list(counts.keys())
    values = [counts[k] for k in keys]

    plt.figure()
    plt.bar(keys, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def maybe_save_current_plot() -> None:
    """
    Ask the user whether to save the current matplotlib figure.
    Validates the chosen output format.
    """
    ans = input("Do you want to save the chart? (y/n): ").strip().lower()
    if ans != "y":
        return

    filename = input("File name (without extension): ").strip()
    fmt = validate_image_format(input("Format (png/jpg/pdf/svg): "))
    out = f"{filename}.{fmt}"
    plt.gcf().savefig(out)
    print(f"Chart saved to: {out}")


def basic_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute min, max, mean, and population standard deviation.
    Returns NaN values if the list is empty.
    """
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}

    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) >= 2 else 0.0
    }


def compute_intervals(values: List[float], n_bins: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Build N intervals between min and max, then count occurrences.
    Interval rule:
      - left edge inclusive
      - right edge exclusive except for last bin (inclusive on both ends)

    Returns:
      intervals: list of (a, b) floats
      counts: list of counts per interval
    """
    if not values:
        return [], []

    vmin = min(values)
    vmax = max(values)

    # Handle constant values (all equal)
    step = (vmax - vmin) / n_bins if vmin != vmax else 0.0

    intervals: List[Tuple[float, float]] = []
    counts = [0] * n_bins

    # Build intervals
    for i in range(n_bins):
        a = vmin + i * step
        b = vmin + (i + 1) * step if i < n_bins - 1 else vmax
        intervals.append((a, b))

    # Count occurrences
    for v in values:
        if step == 0.0:
            counts[-1] += 1
            continue

        # Put exact max into last bin to keep boundaries consistent
        if v == vmax:
            counts[-1] += 1
            continue

        idx = int((v - vmin) / step)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1

    return intervals, counts


def plot_histogram_intervals(intervals: List[Tuple[float, float]], counts: List[int], title: str, xlabel: str) -> None:
    """
    Plot a bar chart representing interval frequencies.
    """
    labels = [f"[{a:.1f}, {b:.1f})" for (a, b) in intervals]
    if labels:
        labels[-1] = labels[-1].replace(")", "]")  # last interval is closed on the right

    plt.figure()
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# =========================
#   KMeans clustering
# =========================

def run_kmeans_and_write_summary(values: List[float], out_path: str, k_min: int = 2, k_max: int = 7) -> None:
    """
    Run KMeans for k in [k_min..k_max] on 1D values (circumference).
    Print cluster sizes and write CSV-text output (no csv module).

    Output columns:
      K,id,nElements,centroidCircumference
    """
    if not values:
        raise ValidationError("Empty list: cannot run KMeans.")

    # sklearn expects 2D input
    X = [[v] for v in values]

    lines = ["K,id,nElements,centroidCircumference"]
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_  # shape: (k, 1)

        # Count elements per cluster
        counts = [0] * k
        for lab in labels:
            counts[int(lab)] += 1

        # Write one line per cluster
        for cid in range(k):
            centroid_val = float(centroids[cid][0])
            lines.append(f"{k},{cid},{counts[cid]},{centroid_val:.4f}")

        print(f"K={k} -> cluster counts:", counts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"CSV text file generated: {out_path}")


# =========================
#   Distance (Haversine)
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance in kilometers between two geographic coordinates.
    """
    validate_lat(lat1); validate_lon(lon1)
    validate_lat(lat2); validate_lon(lon2)

    R = 6371.0  # Earth radius (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def ask_float(prompt: str) -> float:
    """
    Read a float from input(). Supports comma decimal separator by replacing ',' with '.'.
    """
    s = input(prompt).strip().replace(",", ".")
    return float(s)


def choose_species(species_list: List[str]) -> str:
    """
    Let user choose a species among available dataset species.
    The choice must match exactly one entry; otherwise ask again.
    """
    unique = sorted({(s or "").strip() for s in species_list if (s or "").strip() != ""})
    if not unique:
        raise ValidationError("No species available in the dataset.")

    print("\nAvailable species (showing first 30):")
    for s in unique[:30]:
        print("-", s)
    if len(unique) > 30:
        print(f"... (total species: {len(unique)})")

    while True:
        chosen = input("\nType a species (field 'SPECIE NOME VOLGARE'): ").strip()
        if chosen in unique:
            return chosen
        print("Invalid species. Try again (must match exactly one existing species).")


def sort_trees_with_distance(
    trees: List[MonumentalTree],
    user_lat: float,
    user_lon: float,
    sort_mode: str
) -> List[Tuple[MonumentalTree, float]]:
    """
    Compute (tree, distance) pairs and sort them based on user-selected criteria.

    sort_mode options:
      - distance_asc
      - distance_desc
      - circumference_asc
      - circumference_desc
    """
    pairs = [(t, haversine_km(user_lat, user_lon, t.get_lat(), t.get_lon())) for t in trees]

    mode = (sort_mode or "").strip().lower()
    if mode == "distance_asc":
        pairs.sort(key=lambda x: x[1])
    elif mode == "distance_desc":
        pairs.sort(key=lambda x: x[1], reverse=True)
    elif mode == "circumference_asc":
        pairs.sort(key=lambda x: x[0].get_circumference())
    elif mode == "circumference_desc":
        pairs.sort(key=lambda x: x[0].get_circumference(), reverse=True)
    else:
        raise ValidationError("Invalid sort mode.")

    return pairs


# =========================
#   Bounding boxes I/O
# =========================

def parse_bbox_line(line: str) -> Optional[BoundingBox]:
    """
    Parse a bounding box line.
    Accepts separators: ';' or ',' or whitespace.
    Expected 4 values:
      TL_x; TL_y; BR_x; BR_y
    Interpreted as:
      TL_lon; TL_lat; BR_lon; BR_lat

    Returns:
      BoundingBox if valid, otherwise None (invalid boxes must be ignored).
    """
    s = (line or "").strip()
    if not s or s.startswith("#"):
        return None

    # Normalize separators into whitespace
    for sep in [";", ","]:
        s = s.replace(sep, " ")
    parts = [p for p in s.split() if p.strip() != ""]

    if len(parts) != 4:
        return None

    try:
        tl_lon = float(parts[0].replace(",", "."))
        tl_lat = float(parts[1].replace(",", "."))
        br_lon = float(parts[2].replace(",", "."))
        br_lat = float(parts[3].replace(",", "."))

        box = BoundingBox(GeoPoint(tl_lon, tl_lat), GeoPoint(br_lon, br_lat))
        return box if box.is_valid_box() else None
    except Exception:
        return None


def read_bounding_boxes(path: str) -> List[BoundingBox]:
    """
    Read bounding boxes from a file, ignoring invalid ones.
    """
    boxes: List[BoundingBox] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            b = parse_bbox_line(line)
            if b is not None:
                boxes.append(b)
    return boxes


def bbox_stats(trees: List[MonumentalTree], box: BoundingBox) -> Dict[str, object]:
    """
    Compute statistics for trees contained in a bounding box.
    Returns a dictionary ready to be serialized as JSON.
    """
    inside = [t for t in trees if box.contains(t.get_lon(), t.get_lat())]
    heights = [t.get_height() for t in inside]
    circs = [t.get_circumference() for t in inside]

    return {
        "bbox": {
            "top_left": {"lon": box.top_left.lon, "lat": box.top_left.lat},
            "bottom_right": {"lon": box.bottom_right.lon, "lat": box.bottom_right.lat},
        },
        "n_trees": len(inside),
        "height": basic_stats(heights),
        "circumference": basic_stats(circs),
    }


# =========================
#   Dataset loading
# =========================

def to_float_safe(x: str) -> float:
    """
    Convert a string to float safely.
    - Handles comma decimal separator.
    - Returns NaN if empty.
    """
    s = (x or "").strip().replace(",", ".")
    if s == "":
        return math.nan
    return float(s)


def load_trees_from_csv_path(path: str) -> List[MonumentalTree]:
    """
    Load dataset from local CSV path and convert to a list of MonumentalTree objects.

    This uses a custom CSV parser (no csv module).
    Rows with invalid coordinates or missing numeric values for circumference/height are skipped
    to keep analysis robust.
    """
    text = read_text_file(path)
    rows = parse_csv_text(text)

    if not rows:
        raise ValidationError("Dataset is empty or unreadable.")

    header = rows[0]
    data_rows = rows[1:]

    # Build a column index map: column_name -> index
    idx = {name.strip(): i for i, name in enumerate(header)}

    # Columns expected in the dataset
    required = [
        "PROVINCIA",
        "COMUNE",
        "LOCALITÀ",
        "CONTESTO URBANO",
        "SPECIE NOME VOLGARE",
        "CIRCONFERENZA FUSTO (cm)",
        "ALTEZZA (m)",
        "PROPOSTA DICHIARAZIONE NOTEVOLE INTERESSE PUBBLICO",
        "lon",
        "lat",
    ]
    for r in required:
        if r not in idx:
            raise ValidationError(f"Missing column in dataset: {r}")

    trees: List[MonumentalTree] = []

    # Helper to read a column safely even if a row is shorter than expected
    def get_value(row: List[str], col: str) -> str:
        i = idx[col]
        return row[i] if i < len(row) else ""

    for row in data_rows:
        try:
            lon = to_float_safe(get_value(row, "lon"))
            lat = to_float_safe(get_value(row, "lat"))
            validate_lon(lon)
            validate_lat(lat)

            circ = to_float_safe(get_value(row, "CIRCONFERENZA FUSTO (cm)"))
            h = to_float_safe(get_value(row, "ALTEZZA (m)"))

            # circumference and height must be floats, skip rows where they are missing
            if math.isnan(circ) or math.isnan(h):
                continue

            tree = MonumentalTree(
                _lat=lat,
                _lon=lon,
                _province=get_value(row, "PROVINCIA").strip(),
                _urban_context=get_value(row, "CONTESTO URBANO").strip(),
                _species_common_name=get_value(row, "SPECIE NOME VOLGARE").strip(),
                _public_interest_proposal=get_value(row, "PROPOSTA DICHIARAZIONE NOTEVOLE INTERESSE PUBBLICO").strip(),
                _circumference_cm=float(circ),
                _height_m=float(h),
                _town=get_value(row, "COMUNE").strip(),
                _locality=get_value(row, "LOCALITÀ").strip(),
            )

            trees.append(tree)

        except Exception:
            # If a row is malformed or has invalid conversions, ignore it
            continue

    return trees


# =========================
#   Main program (English UI)
# =========================

def main() -> None:
    """
    Main entry point: provides a text-based interface via input().

    The dataset path is set to the mounted location used in this environment.
    If you run locally, put the CSV file in the same folder and change this path accordingly.
    """
    print("=== Data Analysis Project: Monumental Trees of Sicily ===")

    # Default dataset path for the environment used here.
    dataset_path = "alberimonumentalisicilia.csv"

    try:
        trees = load_trees_from_csv_path(dataset_path)
        print(f"\nNumber of available monumental trees: {len(trees)}")

        menu = (
            "\nChoose an action:\n"
            "1) Bar chart by Province\n"
            "2) Bar chart by Urban context\n"
            "3) Bar chart by Public interest proposal\n"
            "4) Intervals + bar charts for Circumference and Height\n"
            "5) Statistics (min/max/mean/std)\n"
            "6) KMeans (k=2..7) on circumference + export CSV text file\n"
            "7) Distances: your position + species + sorting\n"
            "8) Bounding boxes: import + stats + export JSON\n"
            "0) Exit\n"
        )

        while True:
            print(menu)
            choice = input("Your choice: ").strip()

            if choice == "0":
                print("Goodbye.")
                break

            # -----------------------
            # Bar chart by Province
            # -----------------------
            elif choice == "1":
                counts = group_counts(trees, lambda t: t.get_province())
                bar_chart_from_counts(counts, "Monumental Trees by Province", "Province")
                maybe_save_current_plot()

            # ---------------------------
            # Bar chart by Urban context
            # ---------------------------
            elif choice == "2":
                counts = group_counts(trees, lambda t: t.get_urban_context())
                bar_chart_from_counts(counts, "Monumental Trees by Urban Context", "Urban context")
                maybe_save_current_plot()

            # -----------------------------------------
            # Bar chart by Public Interest Proposal field
            # -----------------------------------------
            elif choice == "3":
                counts = group_counts(trees, lambda t: t.get_public_interest_proposal())
                bar_chart_from_counts(counts, "Monumental Trees by Public Interest Proposal", "Proposal")
                maybe_save_current_plot()

            # ---------------------------------------------------
            # Interval frequencies + bar chart for numeric fields
            # ---------------------------------------------------
            elif choice == "4":
                n = int(input("Enter number of intervals N (5..10): ").strip())
                validate_bins(n)

                circs = [t.get_circumference() for t in trees]
                heights = [t.get_height() for t in trees]

                # Circumference intervals
                intervals_c, counts_c = compute_intervals(circs, n)
                print("\nCounts per circumference interval:")
                for (a, b), c in zip(intervals_c, counts_c):
                    print(f"[{a:.2f}, {b:.2f}] -> {c}")

                plot_histogram_intervals(intervals_c, counts_c, "Circumference Intervals", "Circumference (cm)")
                maybe_save_current_plot()

                # Height intervals
                intervals_h, counts_h = compute_intervals(heights, n)
                print("\nCounts per height interval:")
                for (a, b), c in zip(intervals_h, counts_h):
                    print(f"[{a:.2f}, {b:.2f}] -> {c}")

                plot_histogram_intervals(intervals_h, counts_h, "Height Intervals", "Height (m)")
                maybe_save_current_plot()

            # -----------------------
            # Print basic statistics
            # -----------------------
            elif choice == "5":
                circs = [t.get_circumference() for t in trees]
                heights = [t.get_height() for t in trees]

                s_c = basic_stats(circs)
                s_h = basic_stats(heights)

                print("\nCircumference (cm) stats:")
                print(f"min={s_c['min']:.3f}  max={s_c['max']:.3f}  mean={s_c['mean']:.3f}  std={s_c['std']:.3f}")

                print("\nHeight (m) stats:")
                print(f"min={s_h['min']:.3f}  max={s_h['max']:.3f}  mean={s_h['mean']:.3f}  std={s_h['std']:.3f}")

            # -----------------------
            # KMeans clustering
            # -----------------------
            elif choice == "6":
                out_csv = input("Output filename (e.g., kmeans_out.csv): ").strip()
                validate_output_extension(out_csv, ".csv")

                values = [t.get_circumference() for t in trees]
                run_kmeans_and_write_summary(values, out_csv, 2, 7)

            # -----------------------
            # Distances + sorting
            # -----------------------
            elif choice == "7":
                user_lat = ask_float("Enter your latitude: ")
                user_lon = ask_float("Enter your longitude: ")
                validate_lat(user_lat)
                validate_lon(user_lon)

                species = choose_species([t.get_species_common_name() for t in trees])
                filtered = list(filter(lambda t: t.get_species_common_name() == species, trees))

                print(f"\nTrees found for species '{species}': {len(filtered)}")
                if not filtered:
                    continue

                print("\nSorting options:")
                print("1) distance_asc  2) distance_desc  3) circumference_asc  4) circumference_desc")
                opt = input("Choose (1..4): ").strip()
                mapping = {
                    "1": "distance_asc",
                    "2": "distance_desc",
                    "3": "circumference_asc",
                    "4": "circumference_desc"
                }
                sort_mode = mapping.get(opt)
                if sort_mode is None:
                    raise ValidationError("Invalid sorting choice.")

                pairs = sort_trees_with_distance(filtered, user_lat, user_lon, sort_mode)

                limit_s = input("How many results to print? (Enter = all): ").strip()
                limit = int(limit_s) if limit_s else None

                print("\nTrees + distance (km):")
                shown = pairs if limit is None else pairs[:limit]
                for t, d in shown:
                    print(f"{t.short_str()} | distance={d:.3f} km")

            # -----------------------
            # Bounding boxes analysis
            # -----------------------
            elif choice == "8":
                bbox_file = input("Bounding box file name (e.g., bbox.txt): ").strip()
                boxes = read_bounding_boxes(bbox_file)

                print(f"\nValid bounding boxes read: {len(boxes)}")
                if not boxes:
                    print("No valid bounding boxes. Invalid ones were ignored.")
                    continue

                results = []
                for i, b in enumerate(boxes, start=1):
                    st = bbox_stats(trees, b)
                    results.append(st)

                    print(f"\nBOX #{i}:")
                    print("n_trees:", st["n_trees"])
                    print("height:", st["height"])
                    print("circumference:", st["circumference"])

                out_json = input("Output JSON filename (e.g., bbox_out.json): ").strip()
                validate_output_extension(out_json, ".json")

                with open(out_json, "w", encoding="utf-8") as f:
                    f.write(json.dumps(results, ensure_ascii=False, indent=2))

                print(f"\nSaved results to: {out_json}")

            else:
                print("Invalid choice.")

    except ValidationError as ve:
        print(f"[VALIDATION ERROR] {ve}")
    except FileNotFoundError as fe:
        print(f"[FILE ERROR] {fe}")
    except Exception as e:
        print(f"[GENERIC ERROR] {e}")


if __name__ == "__main__":
    main()
