"""A proposed box must survive only when it is one whole item, not a group or a part."""

import unittest

import numpy as np

from shape_recognition import (
    Detection,
    box_area,
    box_containment,
    mask_leak_fraction,
    plausible_boxes,
    suppress_contained_parts,
    suppress_group_boxes,
)


class BoxGeometryTests(unittest.TestCase):
    def test_containment_is_one_when_the_inner_box_lies_wholly_inside(self):
        self.assertEqual(box_containment([10, 10, 20, 20], [0, 0, 100, 100]), 1.0)

    def test_containment_is_zero_for_disjoint_boxes(self):
        self.assertEqual(box_containment([0, 0, 10, 10], [50, 50, 60, 60]), 0.0)

    def test_containment_is_measured_against_the_inner_box_not_the_outer(self):
        # Half of the inner box overlaps, so containment is 0.5 regardless of outer size.
        self.assertAlmostEqual(box_containment([0, 0, 20, 10], [10, 0, 1000, 1000]), 0.5)

    def test_containment_of_a_degenerate_box_is_zero_rather_than_dividing_by_zero(self):
        self.assertEqual(box_containment([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)

    def test_area_of_an_inverted_box_is_zero_rather_than_negative(self):
        self.assertEqual(box_area([20, 20, 10, 10]), 0.0)


class GroupBoxSuppressionTests(unittest.TestCase):
    """One box drawn around two products is the failure this rule exists to remove."""

    def test_a_box_spanning_two_separate_products_is_dropped(self):
        boxes = [[0, 0, 100, 50], [0, 0, 45, 50], [55, 0, 100, 50]]
        scores = [0.25, 0.55, 0.52]
        self.assertEqual(sorted(suppress_group_boxes(boxes, scores)), [1, 2])

    def test_the_individual_products_are_kept_and_never_the_enclosing_box(self):
        boxes = [[0, 0, 100, 50], [0, 0, 45, 50], [55, 0, 100, 50]]
        scores = [0.90, 0.20, 0.19]
        keep = suppress_group_boxes(boxes, scores)
        self.assertNotIn(0, keep, "the spanning box must go even when it scores highest")
        self.assertEqual(sorted(keep), [1, 2])

    def test_a_box_containing_only_one_smaller_box_is_kept(self):
        # Two identical cans stacked and touching produce exactly this shape: a single
        # outer box with one partial box inside it. Keeping the outer box hands a reviewer
        # one box to split rather than silently discarding the item.
        boxes = [[0, 0, 100, 100], [10, 55, 90, 95]]
        scores = [0.37, 0.09]
        self.assertEqual(sorted(suppress_group_boxes(boxes, scores)), [0, 1])

    def test_boxes_that_are_nearly_the_same_size_do_not_count_as_children(self):
        boxes = [[0, 0, 100, 100], [1, 1, 99, 99], [2, 2, 98, 98]]
        scores = [0.5, 0.4, 0.3]
        self.assertEqual(sorted(suppress_group_boxes(boxes, scores)), [0, 1, 2])

    def test_four_products_in_a_row_all_survive_their_spanning_box(self):
        row = [[i * 25, 0, i * 25 + 20, 40] for i in range(4)]
        boxes = [[0, 0, 95, 40]] + row
        scores = [0.24, 0.56, 0.53, 0.52, 0.51]
        self.assertEqual(sorted(suppress_group_boxes(boxes, scores)), [1, 2, 3, 4])

    def test_an_empty_input_returns_no_indices(self):
        self.assertEqual(suppress_group_boxes([], []), [])


class ContainedPartSuppressionTests(unittest.TestCase):
    """A cap or a label inside a stronger box is a part of that item, not a second item."""

    def test_a_box_inside_a_stronger_box_is_dropped(self):
        boxes = [[0, 0, 100, 400], [10, 20, 90, 180]]
        scores = [0.45, 0.12]
        self.assertEqual(suppress_contained_parts(boxes, scores), [0])

    def test_side_by_side_products_that_merely_overlap_are_both_kept(self):
        boxes = [[0, 0, 60, 100], [40, 0, 100, 100]]
        scores = [0.5, 0.48]
        self.assertEqual(sorted(suppress_contained_parts(boxes, scores)), [0, 1])

    def test_the_stronger_box_is_the_one_retained(self):
        boxes = [[10, 20, 90, 180], [0, 0, 100, 400]]
        scores = [0.12, 0.45]
        self.assertEqual(suppress_contained_parts(boxes, scores), [1])

    def test_a_part_of_a_part_is_dropped_only_once(self):
        boxes = [[0, 0, 100, 400], [10, 20, 90, 180], [20, 30, 80, 90]]
        scores = [0.45, 0.30, 0.20]
        self.assertEqual(suppress_contained_parts(boxes, scores), [0])


class PlausibleBoxTests(unittest.TestCase):
    def test_a_box_narrower_than_the_minimum_side_is_rejected(self):
        self.assertEqual(plausible_boxes([[0, 0, 4, 400]], 1000, 1000), [])

    def test_a_box_covering_almost_the_whole_frame_is_rejected_as_the_scene(self):
        self.assertEqual(plausible_boxes([[0, 0, 1000, 1000]], 1000, 1000), [])

    def test_a_speck_below_the_minimum_area_fraction_is_rejected(self):
        self.assertEqual(plausible_boxes([[0, 0, 20, 20]], 1000, 1000), [])

    def test_an_ordinary_product_box_is_kept(self):
        self.assertEqual(plausible_boxes([[100, 100, 400, 500]], 1000, 1000), [0])

    def test_a_zero_sized_frame_does_not_raise(self):
        self.assertEqual(plausible_boxes([[0, 0, 10, 10]], 0, 0), [])


class MaskLeakTests(unittest.TestCase):
    """Leak is how far a mask spilled off its own item and onto a neighbour."""

    def test_a_mask_wholly_inside_its_box_does_not_leak(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 20:40] = True
        self.assertEqual(mask_leak_fraction(mask, [10, 10, 50, 50]), 0.0)

    def test_half_a_mask_outside_the_box_reports_half_the_leak(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:20, 0:20] = True
        self.assertAlmostEqual(mask_leak_fraction(mask, [0, 0, 20, 10]), 0.5)

    def test_an_empty_mask_reports_no_leak_rather_than_dividing_by_zero(self):
        self.assertEqual(mask_leak_fraction(np.zeros((10, 10), dtype=bool), [0, 0, 5, 5]), 0.0)


class DetectionRecordTests(unittest.TestCase):
    def test_a_detection_carries_its_box_score_label_and_optional_mask(self):
        detection = Detection(box=(1.0, 2.0, 3.0, 4.0), score=0.5, label="a jar")
        self.assertIsNone(detection.mask)
        self.assertEqual(detection.box, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(detection.label, "a jar")

    def test_the_mask_is_kept_out_of_the_repr_so_logs_stay_readable(self):
        detection = Detection(box=(0, 0, 1, 1), score=0.1, label="a cup",
                              mask=np.ones((4, 4), dtype=bool))
        self.assertNotIn("True", repr(detection))


if __name__ == "__main__":
    unittest.main()
