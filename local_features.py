"""Conservative label verification for otherwise ambiguous embedding matches."""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


MATCHER_VERSION = "resnet18-sift-ransac-v3"
FEATURE_SIDE = 640
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.35
MIN_COVERAGE = 0.08
GEOMETRY_RESCUE_MIN_SIMILARITY = 0.70
GEOMETRY_RESCUE_MIN_MASK_AREA = 0.005


def should_verify_low_score_geometry(record, maximum_margin):
    """Limit catalogue-wide label checks to sufficiently ambiguous masks."""
    if record.get("reason") != "below_similarity_threshold":
        return False
    gap = record.get("match_margin")
    return bool(record.get("similarity", 0.0) >= GEOMETRY_RESCUE_MIN_SIMILARITY
                and record.get("mask_area_ratio", 0.0) >= GEOMETRY_RESCUE_MIN_MASK_AREA
                and (gap is None or gap <= maximum_margin))


@dataclass
class LocalFeatures:
    points: np.ndarray
    descriptors: np.ndarray | None
    size: tuple


def extract_features(image):
    image = image.convert("RGB")
    foreground = np.any(np.asarray(image) != (124, 124, 124), axis=2).astype(np.uint8) * 255
    image.thumbnail((FEATURE_SIDE, FEATURE_SIDE))
    foreground = cv2.resize(foreground, image.size, interpolation=cv2.INTER_NEAREST)
    foreground = cv2.erode(foreground, np.ones((5, 5), dtype=np.uint8))
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    keys, descriptors = cv2.SIFT_create(nfeatures=1200).detectAndCompute(gray, foreground)
    points = np.float32([key.pt for key in keys]).reshape(-1, 2)
    return LocalFeatures(points, descriptors, image.size)


def verify_geometry(reference, query):
    result = {"verified": False, "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
              "reference_coverage": 0.0, "query_coverage": 0.0}
    if reference.descriptors is None or query.descriptors is None or len(query.descriptors) < 2:
        return result
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference.descriptors, query.descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    selected, source_cells, target_cells = [], set(), set()
    # SIFT can give several orientations at one location; they are not independent evidence.
    for match in sorted(good, key=lambda item: item.distance):
        source = tuple(np.round(reference.points[match.queryIdx] / 2).astype(int))
        target = tuple(np.round(query.points[match.trainIdx] / 2).astype(int))
        if source not in source_cells and target not in target_cells:
            source_cells.add(source)
            target_cells.add(target)
            selected.append(match)
    result["matches"] = len(selected)
    if len(selected) < MIN_INLIERS:
        return result
    source = np.float32([reference.points[item.queryIdx] for item in selected])
    target = np.float32([query.points[item.trainIdx] for item in selected])
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return result
    inliers = inliers.ravel().astype(bool)
    count = int(inliers.sum())
    result.update(inliers=count, inlier_ratio=count / len(selected))
    if count < MIN_INLIERS:
        return result
    result["reference_coverage"] = float(cv2.contourArea(cv2.convexHull(source[inliers]))
                                         / (reference.size[0] * reference.size[1]))
    result["query_coverage"] = float(cv2.contourArea(cv2.convexHull(target[inliers]))
                                     / (query.size[0] * query.size[1]))
    corners = np.float32([[0, 0], [reference.size[0], 0], reference.size,
                          [0, reference.size[1]]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, transform)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return result
    projected_area = cv2.contourArea(projected, oriented=True)
    query_area = query.size[0] * query.size[1]
    result["verified"] = bool(result["inlier_ratio"] >= MIN_INLIER_RATIO
                              and result["reference_coverage"] >= MIN_COVERAGE
                              and result["query_coverage"] >= MIN_COVERAGE
                              and 0.05 * query_area <= projected_area <= 4 * query_area)
    return result


class LabelVerifier:
    def __init__(self):
        self.references = {}

    def compare(self, query, references):
        """Return per-view evidence; callers decide which competing SKUs to verify."""
        query_features = extract_features(query)
        results = []
        for reference, crop in references:
            key = (crop.size, hashlib.sha256(crop.tobytes()).digest())
            if key not in self.references:
                self.references[key] = extract_features(crop)
            results.append({**verify_geometry(self.references[key], query_features),
                            "sku": reference["sku"], "reference_id": reference["reference_id"],
                            "reference_view": reference["view"], "reference_image": reference["image_path"]})
        return results


def resolve_ambiguity(record, evidence, candidate_skus=None):
    """Require one verified SKU, agreeing with the embedding winner; otherwise abstain."""
    if record.get("reason") != "ambiguous_references":
        return
    best_match = record["matches"][0]
    allowed = set(candidate_skus) if candidate_skus is not None else {match["sku"] for match in record["matches"]}
    if {item["sku"] for item in evidence} != allowed:
        raise ValueError("Geometric verification must cover every competing candidate SKU.")
    confirmed = {item["sku"] for item in evidence if item["verified"]}
    verification = {"method": "sift_homography_ransac", "candidates": evidence,
                    "status": "unresolved", "candidate_skus": sorted(allowed)}
    record["geometric_verification"] = verification
    if confirmed != {best_match["sku"]}:
        return
    winner = max((item for item in evidence if item["verified"]), key=lambda item: item["inliers"])
    verification.update(status="verified", **{key: winner[key] for key in
                        ("inliers", "inlier_ratio", "reference_coverage", "query_coverage",
                         "reference_id", "reference_view", "reference_image")})
    record.update(sku=best_match["sku"], status="recognized", reason="geometric_reference_match",
                  embedding_reason="ambiguous_references")


def resolve_low_score_geometry(record, evidence, candidate_skus,
                               minimum_similarity=GEOMETRY_RESCUE_MIN_SIMILARITY,
                               minimum_mask_area=GEOMETRY_RESCUE_MIN_MASK_AREA):
    """Recover a low-scoring mask only when its label uniquely identifies a SKU."""
    if record.get("reason") != "below_similarity_threshold":
        return
    matches = record.get("matches") or []
    if (not matches or matches[0].get("similarity", 0.0) < minimum_similarity
            or record.get("mask_area_ratio", 0.0) < minimum_mask_area):
        return
    allowed = set(candidate_skus)
    if {item["sku"] for item in evidence} != allowed:
        raise ValueError("Geometric recovery must compare every registered SKU.")
    confirmed = {item["sku"] for item in evidence if item["verified"]}
    if len(confirmed) != 1:
        return
    sku = next(iter(confirmed))
    winner = max((item for item in evidence if item["verified"] and item["sku"] == sku),
                 key=lambda item: item["inliers"])
    verification = {"method": "sift_homography_ransac", "candidates": evidence,
                    "status": "verified", "candidate_skus": sorted(allowed),
                    "embedding_winner": matches[0]["sku"]}
    verification.update({key: winner[key] for key in
                         ("inliers", "inlier_ratio", "reference_coverage", "query_coverage",
                          "reference_id", "reference_view", "reference_image")})
    record.update(sku=sku, status="recognized", reason="geometric_reference_match",
                  embedding_reason=record["reason"], geometric_verification=verification)
