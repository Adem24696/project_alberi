"""
test_project_alberi.py
======================

Unit tests for the project.
Requirements:
- At least 8 tests
- Must include at least one assertRaises

Run:
python -m unittest test_project_alberi.py
"""

import unittest

from project_alberi import (
    parse_csv_text,
    haversine_km,
    validate_bins,
    ValidationError,
    MonumentalTree,
    BoundingBox,
    GeoPoint,
    compute_intervals,
    sort_trees_with_distance,
    parse_bbox_line,
)


class TestProjectAlberi(unittest.TestCase):

    def test_parse_csv_simple(self):
        txt = "a,b,c\n1,2,3\n"
        rows = parse_csv_text(txt)
        self.assertEqual(rows[0], ["a", "b", "c"])
        self.assertEqual(rows[1], ["1", "2", "3"])

    def test_parse_csv_quotes_and_newline(self):
        txt = 'a,b\n"hello\nworld",2\n'
        rows = parse_csv_text(txt)
        self.assertEqual(rows[1][0], "hello\nworld")
        self.assertEqual(rows[1][1], "2")

    def test_haversine_zero_distance(self):
        d = haversine_km(37.0, 13.0, 37.0, 13.0)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_validate_bins_ok(self):
        validate_bins(5)
        validate_bins(10)

    def test_validate_bins_raises(self):
        # Required assertRaises example
        with self.assertRaises(ValidationError):
            validate_bins(4)

    def test_tree_float_fields(self):
        t = MonumentalTree(
            _lat=37.0,
            _lon=13.0,
            _province="Agrigento",
            _urban_context="no",
            _species_common_name="Olive",
            _public_interest_proposal="YES",
            _circumference_cm=380.0,
            _height_m=5.0,
            _town="Agrigento",
            _locality="Valley"
        )
        self.assertIsInstance(t.get_circumference(), float)
        self.assertIsInstance(t.get_height(), float)

    def test_bounding_box_contains(self):
        box = BoundingBox(
            top_left=GeoPoint(9.0, 46.0),
            bottom_right=GeoPoint(10.0, 45.0),
        )
        self.assertTrue(box.is_valid_box())
        self.assertTrue(box.contains(9.5, 45.5))
        self.assertFalse(box.contains(11.0, 45.5))

    def test_compute_intervals_counts(self):
        values = [0.0, 1.0, 2.0, 3.0]
        intervals, counts = compute_intervals(values, 2)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(sum(counts), 4)

    def test_sort_modes(self):
        t1 = MonumentalTree(37.0, 13.0, "A", "no", "Olive", "YES", 300.0, 10.0)
        t2 = MonumentalTree(37.1, 13.1, "A", "no", "Olive", "YES", 100.0, 8.0)
        trees = [t1, t2]

        pairs = sort_trees_with_distance(trees, 37.0, 13.0, "circumference_asc")
        self.assertEqual(pairs[0][0].get_circumference(), 100.0)

        pairs2 = sort_trees_with_distance(trees, 37.0, 13.0, "circumference_desc")
        self.assertEqual(pairs2[0][0].get_circumference(), 300.0)

    def test_parse_bbox_line_invalid(self):
        # Invalid because it does not have 4 values
        self.assertIsNone(parse_bbox_line("9.1;45.4;9.2"))

    def test_parse_bbox_line_valid(self):
        # Valid TL lon < BR lon and TL lat > BR lat
        b = parse_bbox_line("9.14;45.48;9.15;45.38")
        self.assertIsNotNone(b)
        self.assertTrue(b.is_valid_box())


if __name__ == "__main__":
    unittest.main()
