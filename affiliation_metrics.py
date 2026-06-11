"""
Affiliation-based precision/recall/F1 for time-series anomaly detection.

Adapted from the MIT-licensed reference implementation:
https://github.com/ahstat/affiliation-metrics-py

This module keeps the implementation self-contained so the repo can compute
affiliation metrics without an additional runtime dependency.
"""

import math
from typing import Iterable, List, Optional, Sequence, Tuple


Interval = Tuple[float, float]


def _sum_without_nan(values: Iterable[float]) -> float:
    return sum(value for value in values if not math.isnan(value))


def _count_without_nan(values: Iterable[float]) -> int:
    return sum(0 if math.isnan(value) else 1 for value in values)


def interval_length(interval: Optional[Interval]) -> float:
    if interval is None:
        return 0.0
    return float(interval[1] - interval[0])


def sum_interval_lengths(intervals: Sequence[Optional[Interval]]) -> float:
    return sum(interval_length(interval) for interval in intervals)


def interval_intersection(
    left: Optional[Interval],
    right: Optional[Interval],
) -> Optional[Interval]:
    if left is None or right is None:
        return None
    start = max(left[0], right[0])
    stop = min(left[1], right[1])
    if start >= stop:
        return None
    return (start, stop)


def interval_is_subset(inner: Interval, outer: Interval) -> bool:
    return inner[0] >= outer[0] and inner[1] <= outer[1]


def split_interval(interval: Optional[Interval], pivot: Interval):
    """
    Split `interval` into [before pivot, overlap, after pivot].
    """
    if interval is None:
        return (None, None, None)

    overlap = interval_intersection(interval, pivot)
    if overlap == interval:
        return (None, overlap, None)
    if interval[1] <= pivot[0]:
        return (interval, None, None)
    if interval[0] >= pivot[1]:
        return (None, None, interval)
    if interval[0] <= pivot[0] and interval[1] >= pivot[1]:
        return ((interval[0], overlap[0]), overlap, (overlap[1], interval[1]))
    if interval[0] <= pivot[0]:
        return ((interval[0], overlap[0]), overlap, None)
    if interval[1] >= pivot[1]:
        return (None, overlap, (overlap[1], interval[1]))
    raise ValueError("Unexpected interval split case.")


def closest_point(outside: Interval, reference: Interval) -> float:
    if interval_intersection(outside, reference) is not None:
        raise ValueError("Intervals must be disjoint when computing pivot.")
    if max(outside) <= min(reference):
        return float(min(reference))
    if min(outside) >= max(reference):
        return float(max(reference))
    raise ValueError("Expected the first interval to lie fully outside the second.")


def convert_vector_to_events(vector: Sequence[int]) -> List[Interval]:
    """
    Convert a binary vector into half-open anomaly intervals [start, stop).
    """
    events: List[Interval] = []
    start = None
    for idx, value in enumerate(vector):
        is_positive = int(value) > 0
        if is_positive and start is None:
            start = idx
        elif not is_positive and start is not None:
            events.append((float(start), float(idx)))
            start = None
    if start is not None:
        events.append((float(start), float(len(vector))))
    return events


def infer_time_range(events_pred: Sequence[Interval], events_gt: Sequence[Interval]) -> Interval:
    if not events_gt:
        raise ValueError("Ground-truth events must not be empty.")
    if not events_pred:
        first = min(event[0] for event in events_gt)
        last = max(event[1] for event in events_gt)
        return (first, last)
    starts = [event[0] for event in events_pred] + [event[0] for event in events_gt]
    stops = [event[1] for event in events_pred] + [event[1] for event in events_gt]
    return (min(starts), max(stops))


def validate_events(events: Sequence[Interval]) -> None:
    if not isinstance(events, list):
        raise TypeError("Events must be provided as a list of (start, stop) tuples.")
    for event in events:
        if not isinstance(event, tuple) or len(event) != 2:
            raise TypeError("Each event must be a (start, stop) tuple.")
        if event[0] > event[1]:
            raise ValueError("Event start must be <= stop.")
    for idx in range(len(events) - 1):
        if events[idx][1] >= events[idx + 1][0]:
            raise ValueError("Events must be ordered and disjoint.")


def affiliation_zones(events_gt: Sequence[Interval], time_range: Interval) -> List[Interval]:
    zones = []
    for idx, current in enumerate(events_gt):
        if idx == 0:
            left = float(time_range[0])
        else:
            left = (events_gt[idx - 1][1] + current[0]) / 2.0
        if idx == len(events_gt) - 1:
            right = float(time_range[1])
        else:
            right = (current[1] + events_gt[idx + 1][0]) / 2.0
        zones.append((left, right))
    return zones


def partition_by_zone(
    events_pred: Sequence[Interval],
    zones: Sequence[Interval],
) -> List[List[Optional[Interval]]]:
    return [
        [interval_intersection(event, zone) for event in events_pred]
        for zone in zones
    ]


def integral_disjoint_distance(interval: Optional[Interval], reference: Interval) -> float:
    if interval is None:
        return 0.0
    pivot = closest_point(interval, reference)
    start = min(interval)
    stop = max(interval)
    return (stop - start) * abs(pivot - (start + stop) / 2.0)


def integral_interval_distance(interval: Optional[Interval], reference: Interval) -> float:
    before, overlap, after = split_interval(interval, reference)
    return (
        integral_disjoint_distance(before, reference)
        + 0.0 * interval_length(overlap)
        + integral_disjoint_distance(after, reference)
    )


def integral_precision_min_piece(interval: Interval, event_gt: Interval, zone: Interval) -> float:
    if interval_intersection(interval, event_gt) is not None:
        raise ValueError("The disjoint precision helper expects non-overlapping intervals.")
    if not interval_is_subset(event_gt, zone):
        raise ValueError("Ground-truth event must lie within its affiliation zone.")
    if not interval_is_subset(interval, zone):
        raise ValueError("Predicted interval must lie within the affiliation zone.")

    zone_start, zone_stop = min(zone), max(zone)
    gt_start, gt_stop = min(event_gt), max(event_gt)
    pred_start, pred_stop = min(interval), max(interval)

    d_min = max(pred_start - gt_stop, gt_start - pred_stop)
    d_max = max(pred_stop - gt_stop, gt_start - pred_start)
    cap = min(gt_start - zone_start, zone_stop - gt_stop)
    area = min(d_max, cap) ** 2 - min(d_min, cap) ** 2
    tail = max(d_max, cap) - max(d_min, cap)
    return 0.5 * area + cap * tail


def integral_precision_disjoint(interval: Interval, event_gt: Interval, zone: Interval) -> float:
    min_piece = integral_precision_min_piece(interval, event_gt, zone)

    zone_start, zone_stop = min(zone), max(zone)
    gt_start, gt_stop = min(event_gt), max(event_gt)
    pred_start, pred_stop = min(interval), max(interval)

    d_min = max(pred_start - gt_stop, gt_start - pred_stop)
    d_max = max(pred_stop - gt_stop, gt_start - pred_start)
    linear_piece = 0.5 * (d_max ** 2 - d_min ** 2)
    overlap_piece = (gt_stop - gt_start) * (pred_stop - pred_start)
    interval_span = pred_stop - pred_start
    zone_span = zone_stop - zone_start
    return interval_span - (min_piece + linear_piece + overlap_piece) / zone_span


def integral_precision_probability(
    interval: Optional[Interval],
    event_gt: Interval,
    zone: Interval,
) -> float:
    before, overlap, after = split_interval(interval, event_gt)

    def score(piece: Optional[Interval]) -> float:
        if piece is None:
            return 0.0
        return integral_precision_disjoint(piece, event_gt, zone)

    overlap_score = 0.0 if overlap is None else (max(overlap) - min(overlap))
    return score(before) + overlap_score + score(after)


def split_by_center(interval: Optional[Interval], center: float):
    if interval is None:
        return (None, None)
    if center >= max(interval):
        return (interval, None)
    if center <= min(interval):
        return (None, interval)
    return ((min(interval), center), (center, max(interval)))


def integral_recall_disjoint(interval_pred: Interval, interval_gt: Interval, zone: Interval) -> float:
    pivot = closest_point(interval_gt, interval_pred)
    zone_start, zone_stop = min(zone), max(zone)
    zone_center = (zone_start + zone_stop) / 2.0

    if pivot <= zone_start or pivot >= zone_stop:
        return 0.0

    before_center, after_center = split_by_center(interval_gt, zone_center)

    left_mid = (zone_start + pivot) / 2.0
    before_far, before_near = split_by_center(before_center, left_mid)

    right_mid = (zone_stop + pivot) / 2.0
    after_near, after_far = split_by_center(after_center, right_mid)

    def bounds(interval: Optional[Interval]):
        if interval is None:
            return (math.nan, math.nan)
        return (min(interval), max(interval))

    bb_min, bb_max = bounds(before_far)
    bn_min, bn_max = bounds(before_near)
    an_min, an_max = bounds(after_near)
    af_min, af_max = bounds(after_far)

    if pivot >= max(interval_gt):
        parts = [
            (pivot - zone_start) * (bb_max - bb_min),
            2 * pivot * (bn_max - bn_min) - (bn_max ** 2 - bn_min ** 2),
            2 * pivot * (an_max - an_min) - (an_max ** 2 - an_min ** 2),
            (zone_stop + pivot) * (af_max - af_min) - (af_max ** 2 - af_min ** 2),
        ]
    elif pivot <= min(interval_gt):
        parts = [
            (bb_max ** 2 - bb_min ** 2) - (zone_start + pivot) * (bb_max - bb_min),
            (bn_max ** 2 - bn_min ** 2) - 2 * pivot * (bn_max - bn_min),
            (an_max ** 2 - an_min ** 2) - 2 * pivot * (an_max - an_min),
            (zone_stop - pivot) * (af_max - af_min),
        ]
    else:
        raise ValueError("Expected the pivot to lie outside the ground-truth slice.")

    integral_min_plus_distance = _sum_without_nan(parts)
    gt_span = max(interval_gt) - min(interval_gt)
    zone_span = zone_stop - zone_start
    return gt_span - integral_min_plus_distance / zone_span


def integral_recall_probability(
    interval_pred: Interval,
    interval_gt: Interval,
    zone: Interval,
) -> float:
    before, overlap, after = split_interval(interval_gt, interval_pred)

    def score(piece: Optional[Interval]) -> float:
        if piece is None:
            return 0.0
        return integral_recall_disjoint(interval_pred, piece, zone)

    overlap_score = 0.0 if overlap is None else (max(overlap) - min(overlap))
    return score(before) + overlap_score + score(after)


def affiliation_precision_probability(
    partitioned_preds: Sequence[Optional[Interval]],
    event_gt: Interval,
    zone: Interval,
) -> float:
    if all(piece is None for piece in partitioned_preds):
        return math.nan
    total_length = sum_interval_lengths(partitioned_preds)
    total_score = sum(
        integral_precision_probability(piece, event_gt, zone)
        for piece in partitioned_preds
        if piece is not None
    )
    return total_score / total_length


def affiliation_recall_probability(
    partitioned_preds: Sequence[Optional[Interval]],
    event_gt: Interval,
    zone: Interval,
) -> float:
    predicted_pieces = [piece for piece in partitioned_preds if piece is not None]
    if not predicted_pieces:
        return 0.0

    recall_zones = affiliation_zones(predicted_pieces, zone)
    ground_truth_slices = partition_by_zone([event_gt], recall_zones)
    total_score = sum(
        integral_recall_probability(pred_piece, gt_slices[0], zone)
        for pred_piece, gt_slices in zip(predicted_pieces, ground_truth_slices)
    )
    return total_score / interval_length(event_gt)


def pr_from_events(
    events_pred: List[Interval],
    events_gt: List[Interval],
    time_range: Optional[Interval],
):
    validate_events(events_pred)
    validate_events(events_gt)

    if not events_gt:
        raise ValueError("Ground-truth events must not be empty.")

    inferred = infer_time_range(events_pred, events_gt)
    if time_range is None:
        raise ValueError("time_range must be provided for affiliation metrics.")
    if time_range[0] > inferred[0] or time_range[1] < inferred[1]:
        raise ValueError("time_range must cover both predicted and ground-truth events.")

    zones = affiliation_zones(events_gt, time_range)
    partitions = partition_by_zone(events_pred, zones)

    precision_terms = [
        affiliation_precision_probability(pred_partition, gt_event, zone)
        for pred_partition, gt_event, zone in zip(partitions, events_gt, zones)
    ]
    recall_terms = [
        affiliation_recall_probability(pred_partition, gt_event, zone)
        for pred_partition, gt_event, zone in zip(partitions, events_gt, zones)
    ]

    if _count_without_nan(precision_terms) > 0:
        precision = _sum_without_nan(precision_terms) / _count_without_nan(precision_terms)
    else:
        precision = math.nan
    recall = sum(recall_terms) / len(recall_terms)
    return {"precision": precision, "recall": recall}


def affiliation_metrics_from_binary_vectors(
    y_pred: Sequence[int],
    y_true: Sequence[int],
):
    """
    Compute affiliation precision/recall/F1 from binary predictions and labels.
    """
    pred_events = convert_vector_to_events(y_pred)
    true_events = convert_vector_to_events(y_true)
    if not true_events:
        return {"precision": math.nan, "recall": math.nan, "f1": math.nan}

    metrics = pr_from_events(pred_events, true_events, (0.0, float(len(y_true))))
    precision = metrics["precision"]
    recall = metrics["recall"]

    if math.isnan(precision):
        precision = 0.0
    if math.isnan(recall):
        recall = 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
