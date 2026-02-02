"""
project_alberi.py
=================

Data Analysis Project (Intro to Python Programming)
Dataset: Monumental Trees of Sicily

Key constraints
---------------
- No pandas
- No csv module
- KMeans must be from sklearn
- Model dataset rows using classes, including inheritance for geographic data
- Model bounding boxes using classes, using inheritance/composition
- Must validate user inputs and handle execution exceptions

Additional requirements implemented
-----------------------------------
- All saved/generated files (images, CSV text file, JSON) are saved in ONE directory: ./output/
- If user enters an invalid filename or format, the program REPROMPTS (does not exit)

Plot saving fix
---------------
Plots were previously saved as blank/white because plt.show() was called before saving.
This version:
- Plot functions RETURN a Figure object and DO NOT call plt.show().
- main() saves via fig.savefig(...) before plt.show(), then closes the figure.
"""

import math
import json
import os
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# =========================
#   Configuration
# =========================

OUTPUT_DIR = "output"  # all generated files go here


def ensure_output_dir() -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def out_path(filename: str) -> str:
    """Join filename into the output directory."""
    return os.path.join(OUTPUT_DIR, filename)


# =========================
#   Exceptions / Validation
# =========================

class ValidationError(ValueError):
    """Raised when input or data fails validation."""
    pass


def validate_lat(lat: float) -> None:
    if not isinstance(lat, (int, float)) or math.isnan(lat) or lat < -90 or lat > 90:
        raise ValidationError("Invalid latitude (must be between -90 and 90).")


def validate_lon(lon: float) -> None:
    if not isinstance(lon, (int, float)) or math.isnan(lon) or lon < -180 or lon > 180:
        raise ValidationError("Invalid longitude (must be between -180 and 180).")


def validate_bins(n: int) -> None:
    if not isinstance(n, int):
        raise ValidationError("N must be an integer.")
    if n < 5 or n > 10:
        raise ValidationError("N must be between 5 and 10 (inclusive).")


def validate_image_format(fmt: str) -> str:
    fmt = (fmt or "").strip().lower().lstrip(".")
    allowed = {"png", "jpg", "jpeg", "pdf", "svg"}
    if fmt not in allowed:
        raise ValidationError(f"Invalid format. Choose one of: {', '.join(sorted(allowed))}")
    if fmt == "jpeg":
        fmt = "jpg"
    return fmt


def validate_output_extension(filename: str, ext: str) -> None:
    if not filename.lower().endswith(ext.lower()):
        raise ValidationError(f"File must end with {ext}")


def nonempty_filename(name: str) -> None:
    if not name or not name.strip():
        raise ValidationError("Filename cannot be empty.")


# =========================
#   Safe input helpers
# =========================

def prompt_int_in_range(prompt: str, min_v: int, max_v: int) -> int:
    """Reprompt until user enters an integer in [min_v, max_v]."""
    while True:
        s = input(prompt).strip()
        try:
            n = int(s)
            if n < min_v or n > max_v:
                print(f"Please enter an integer between {min_v} and {max_v}.")
                continue
            return n
        except ValueError:
            print("Invalid number. Please enter an integer.")


def prompt_float(prompt: str) -> float:
    """Reprompt until user enters a valid float (supports comma as decimal separator)."""
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("Invalid number. Please enter a valid numeric value.")


def prompt_existing_file(prompt: str) -> str:
    """Reprompt until the user provides a path to an existing file."""
    while True:
        path = input(prompt).strip()
        if os.path.isfile(path):
            return path
        print("File not found. Please enter a valid existing file path/name.")


def prompt_output_filename(ext: str, prompt: str) -> str:
    """
    Reprompt until the user provides a non-empty filename with correct extension.
    Returned path is placed inside OUTPUT_DIR.
    """
    while True:
        name = input(prompt).strip()
        try:
            nonempty_filename(name)
            validate_output_extension(name, ext)
            return out_path(name)
        except ValidationError as e:
            print(f"Invalid filename: {e}")


def prompt_plot_save_path() -> Optional[str]:
    """
    Ask user if they want to save the current plot.
    If yes, reprompt until valid filename and format are provided.
    Returns full output path or None.
    """
    ans = input("Do you want to save the chart? (y/n): ").strip().lower()
    if ans != "y":
        return None

    while True:
        base = input("File name (without extension): ").strip()
        fmt = input("Format (png/jpg/pdf/svg): ").strip()
        try:
            nonempty_filename(base)
            fmt = validate_image_format(fmt)
            return out_path(f"{base}.{fmt}")
        except ValidationError as e:
            print(f"Invalid input: {e}")


# =================================
#   CSV Parser (NO csv module)
# =================================

def parse_csv_text(text: str, delimiter: str = ",", quote: str = '"') -> List[List[str]]:
    rows: List[List[str]] = []
    current_row: List[str] = []
    current_field_chars: List[str] = []
    in_quotes = False

    i = 0
    while i < len(text):
        ch = text[i]

        if ch == quote:
            if in_quotes and i + 1 < len(text) and text[i + 1] == quote:
                current_field_chars.append(quote)
                i += 1
            else:
                in_quotes = not in_quotes

        elif ch == delimiter and not in_quotes:
            current_row.append("".join(current_field_chars).strip())
            current_field_chars = []

        elif ch == "\n" and not in_quotes:
            current_row.append("".join(current_field_chars).strip())
            current_field_chars = []
            if not (len(current_row) == 1 and current_row[0] == ""):
                rows.append(current_row)
            current_row = []

        else:
            current_field_chars.append(ch)

        i += 1

    if current_field_chars or current_row:
        current_row.append("".join(current_field_chars).strip())
        if not (len(current_row) == 1 and current_row[0] == ""):
            rows.append(current_row)

    return rows


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


# =========================
#   Domain model (Classes)
# =========================

@dataclass
class GeoEntity:
    _lat: float
    _lon: float

    def __post_init__(self):
        validate_lat(self._lat)
        validate_lon(self._lon)

    def get_lat(self) -> float:
        return self._lat

    def get_lon(self) -> float:
        return self._lon

    def set_lat(self, lat: float) -> None:
        validate_lat(lat)
        self._lat = float(lat)

    def set_lon(self, lon: float) -> None:
        validate_lon(lon)
        self._lon = float(lon)


@dataclass
class MonumentalTree(GeoEntity):
    _province: str
    _urban_context: str
    _species_common_name: str
    _public_interest_proposal: str
    _circumference_cm: float
    _height_m: float
    _town: str = ""
    _locality: str = ""

    def get_province(self) -> str:
        return self._province

    def get_urban_context(self) -> str:
        return self._urban_context

    def get_species_common_name(self) -> str:
        return self._species_common_name

    def get_public_interest_proposal(self) -> str:
        return self._public_interest_proposal

    def get_circumference(self) -> float:
        return self._circumference_cm

    def get_height(self) -> float:
        return self._height_m

    def short_str(self) -> str:
        return (
            f"{self._species_common_name} | Province: {self._province} | "
            f"Town: {self._town} | Circ: {self._circumference_cm:.1f} cm | "
            f"Height: {self._height_m:.1f} m"
        )


@dataclass
class GeoPoint:
    lon: float
    lat: float

    def __post_init__(self):
        validate_lon(self.lon)
        validate_lat(self.lat)


@dataclass
class BoundingBox:
    top_left: GeoPoint
    bottom_right: GeoPoint

    def is_valid_box(self) -> bool:
        return (self.top_left.lon < self.bottom_right.lon) and (self.top_left.lat > self.bottom_right.lat)

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.top_left.lon <= lon <= self.bottom_right.lon and
            self.bottom_right.lat <= lat <= self.top_left.lat
        )


# =========================
#   Analytics functions
# =========================

def group_counts(items: List[MonumentalTree], key_fn) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in items:
        k = (key_fn(it) or "").strip() or "NOT SPECIFIED"
        counts[k] = counts.get(k, 0) + 1
    return counts


def bar_chart_from_counts(counts: Dict[str, int], title: str, xlabel: str, ylabel: str = "Count"):
    """
    Create a bar chart and return the matplotlib Figure object.

    Important:
      - This function does NOT call plt.show().
      - Saving should be done with fig.savefig(...) before plt.show() is called.
    """
    keys = list(counts.keys())
    values = [counts[k] for k in keys]

    fig = plt.figure()
    plt.bar(keys, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def basic_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) >= 2 else 0.0
    }


def compute_intervals(values: List[float], n_bins: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    if not values:
        return [], []

    vmin = min(values)
    vmax = max(values)
    step = (vmax - vmin) / n_bins if vmin != vmax else 0.0

    intervals: List[Tuple[float, float]] = []
    counts = [0] * n_bins

    for i in range(n_bins):
        a = vmin + i * step
        b = vmin + (i + 1) * step if i < n_bins - 1 else vmax
        intervals.append((a, b))

    for v in values:
        if step == 0.0 or v == vmax:
            counts[-1] += 1
            continue
        idx = int((v - vmin) / step)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1

    return intervals, counts


def plot_histogram_intervals(intervals: List[Tuple[float, float]], counts: List[int], title: str, xlabel: str):
    """
    Create an interval-frequency bar chart and return the matplotlib Figure object.

    Important:
      - This function does NOT call plt.show().
      - Saving should be done with fig.savefig(...) before plt.show() is called.
    """
    labels = [f"[{a:.1f}, {b:.1f})" for (a, b) in intervals]
    if labels:
        labels[-1] = labels[-1].replace(")", "]")

    fig = plt.figure()
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# =========================
#   KMeans clustering
# =========================

def run_kmeans_and_write_summary(values: List[float], output_csv_path: str, k_min: int = 2, k_max: int = 7) -> None:
    if not values:
        raise ValidationError("Empty list: cannot run KMeans.")

    X = [[v] for v in values]
    lines = ["K,id,nElements,centroidCircumference"]

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_

        counts = [0] * k
        for lab in labels:
            counts[int(lab)] += 1

        for cid in range(k):
            centroid_val = float(centroids[cid][0])
            lines.append(f"{k},{cid},{counts[cid]},{centroid_val:.4f}")

        print(f"K={k} -> cluster counts:", counts)

    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"CSV saved to: {output_csv_path}")


# =========================
#   Distance (Haversine)
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    validate_lat(lat1); validate_lon(lon1)
    validate_lat(lat2); validate_lon(lon2)

    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def choose_species(species_list: List[str]) -> str:
    unique = sorted({(s or "").strip() for s in species_list if (s or "").strip()})
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
#   Bounding boxes
# =========================

def parse_bbox_line(line: str) -> Optional[BoundingBox]:
    s = (line or "").strip()
    if not s or s.startswith("#"):
        return None

    for sep in [";", ","]:
        s = s.replace(sep, " ")
    parts = [p for p in s.split() if p.strip()]

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
    boxes: List[BoundingBox] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            b = parse_bbox_line(line)
            if b is not None:
                boxes.append(b)
    return boxes


def bbox_stats(trees: List[MonumentalTree], box: BoundingBox) -> Dict[str, object]:
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
    s = (x or "").strip().replace(",", ".")
    if s == "":
        return math.nan
    return float(s)


def load_trees_from_csv_path(path: str) -> List[MonumentalTree]:
    text = read_text_file(path)
    rows = parse_csv_text(text)
    if not rows:
        raise ValidationError("Dataset is empty or unreadable.")

    header = rows[0]
    data_rows = rows[1:]
    idx = {name.strip(): i for i, name in enumerate(header)}

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

    def get_value(row: List[str], col: str) -> str:
        i = idx[col]
        return row[i] if i < len(row) else ""

    trees: List[MonumentalTree] = []
    for row in data_rows:
        try:
            lon = to_float_safe(get_value(row, "lon"))
            lat = to_float_safe(get_value(row, "lat"))
            validate_lon(lon)
            validate_lat(lat)

            circ = to_float_safe(get_value(row, "CIRCONFERENZA FUSTO (cm)"))
            h = to_float_safe(get_value(row, "ALTEZZA (m)"))
            if math.isnan(circ) or math.isnan(h):
                continue

            trees.append(
                MonumentalTree(
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
            )
        except Exception:
            continue

    return trees


# =========================
#   Main (English text UI)
# =========================

def main() -> None:
    ensure_output_dir()

    print("=== Data Analysis Project: Monumental Trees of Sicily ===")
    print(f"All generated files will be saved in: ./{OUTPUT_DIR}/")

    dataset_path = "alberimonumentalisicilia.csv"

    # Dataset loading: if missing, show error and stop
    try:
        trees = load_trees_from_csv_path(dataset_path)
    except Exception as e:
        print(f"[DATASET ERROR] {e}")
        return

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

        try:
            if choice == "0":
                print("Goodbye.")
                break

            elif choice == "1":
                counts = group_counts(trees, lambda t: t.get_province())
                fig = bar_chart_from_counts(counts, "Monumental Trees by Province", "Province")

                save_path = prompt_plot_save_path()
                if save_path:
                    fig.savefig(save_path)
                    print(f"Chart saved to: {save_path}")

                plt.show()
                plt.close(fig)

            elif choice == "2":
                counts = group_counts(trees, lambda t: t.get_urban_context())
                fig = bar_chart_from_counts(counts, "Monumental Trees by Urban Context", "Urban context")

                save_path = prompt_plot_save_path()
                if save_path:
                    fig.savefig(save_path)
                    print(f"Chart saved to: {save_path}")

                plt.show()
                plt.close(fig)

            elif choice == "3":
                counts = group_counts(trees, lambda t: t.get_public_interest_proposal())
                fig = bar_chart_from_counts(counts, "Monumental Trees by Public Interest Proposal", "Proposal")

                save_path = prompt_plot_save_path()
                if save_path:
                    fig.savefig(save_path)
                    print(f"Chart saved to: {save_path}")

                plt.show()
                plt.close(fig)

            elif choice == "4":
                n = prompt_int_in_range("Enter number of intervals N (5..10): ", 5, 10)

                circs = [t.get_circumference() for t in trees]
                heights = [t.get_height() for t in trees]

                intervals_c, counts_c = compute_intervals(circs, n)
                print("\nCounts per circumference interval:")
                for (a, b), c in zip(intervals_c, counts_c):
                    print(f"[{a:.2f}, {b:.2f}] -> {c}")

                fig1 = plot_histogram_intervals(intervals_c, counts_c, "Circumference Intervals", "Circumference (cm)")
                save_path = prompt_plot_save_path()
                if save_path:
                    fig1.savefig(save_path)
                    print(f"Chart saved to: {save_path}")
                plt.show()
                plt.close(fig1)

                intervals_h, counts_h = compute_intervals(heights, n)
                print("\nCounts per height interval:")
                for (a, b), c in zip(intervals_h, counts_h):
                    print(f"[{a:.2f}, {b:.2f}] -> {c}")

                fig2 = plot_histogram_intervals(intervals_h, counts_h, "Height Intervals", "Height (m)")
                save_path = prompt_plot_save_path()
                if save_path:
                    fig2.savefig(save_path)
                    print(f"Chart saved to: {save_path}")
                plt.show()
                plt.close(fig2)

            elif choice == "5":
                circs = [t.get_circumference() for t in trees]
                heights = [t.get_height() for t in trees]
                sc = basic_stats(circs)
                sh = basic_stats(heights)

                print("\nCircumference (cm) stats:")
                print(f"min={sc['min']:.3f}  max={sc['max']:.3f}  mean={sc['mean']:.3f}  std={sc['std']:.3f}")

                print("\nHeight (m) stats:")
                print(f"min={sh['min']:.3f}  max={sh['max']:.3f}  mean={sh['mean']:.3f}  std={sh['std']:.3f}")

            elif choice == "6":
                output_csv = prompt_output_filename(".csv", "Output filename (must end with .csv): ")
                values = [t.get_circumference() for t in trees]
                run_kmeans_and_write_summary(values, output_csv, 2, 7)

            elif choice == "7":
                user_lat = prompt_float("Enter your latitude: ")
                user_lon = prompt_float("Enter your longitude: ")
                validate_lat(user_lat)
                validate_lon(user_lon)

                species = choose_species([t.get_species_common_name() for t in trees])
                filtered = list(filter(lambda t: t.get_species_common_name() == species, trees))

                print(f"\nTrees found for species '{species}': {len(filtered)}")
                if not filtered:
                    continue

                print("\nSorting options:")
                print("1) distance_asc  2) distance_desc  3) circumference_asc  4) circumference_desc")
                while True:
                    opt = input("Choose (1..4): ").strip()
                    mapping = {"1": "distance_asc", "2": "distance_desc", "3": "circumference_asc", "4": "circumference_desc"}
                    sort_mode = mapping.get(opt)
                    if sort_mode:
                        break
                    print("Invalid choice. Please select 1..4.")

                pairs = sort_trees_with_distance(filtered, user_lat, user_lon, sort_mode)

                limit_s = input("How many results to print? (Enter = all): ").strip()
                limit = None
                if limit_s:
                    try:
                        limit = int(limit_s)
                        if limit <= 0:
                            limit = None
                    except ValueError:
                        limit = None

                print("\nTrees + distance (km):")
                shown = pairs if limit is None else pairs[:limit]
                for t, d in shown:
                    print(f"{t.short_str()} | distance={d:.3f} km")

            elif choice == "8":
                bbox_file = prompt_existing_file("Bounding box file name (must exist): ")
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

                output_json = prompt_output_filename(".json", "Output JSON filename (must end with .json): ")
                with open(output_json, "w", encoding="utf-8") as f:
                    f.write(json.dumps(results, ensure_ascii=False, indent=2))
                print(f"\nJSON saved to: {output_json}")

            else:
                print("Invalid choice.")

        except ValidationError as ve:
            print(f"[VALIDATION ERROR] {ve}")
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
