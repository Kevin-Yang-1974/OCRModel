from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


DIRECTION_LABELS = (
    "horizontal_ltr",
    "horizontal_rtl",
    "vertical_rtl",
    "vertical_ltr",
    "unknown",
)


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def levenshtein_distance(reference: str, prediction: str) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_char in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + (reference_char != prediction_char),
                )
            )
        previous = current
    return previous[-1]


def remove_whitespace(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def evaluate_ocr_page(reference_text: str, predicted_text: str) -> dict[str, Any]:
    edit_distance = levenshtein_distance(reference_text, predicted_text)
    reference_without_whitespace = remove_whitespace(reference_text)
    prediction_without_whitespace = remove_whitespace(predicted_text)
    whitespace_edit_distance = levenshtein_distance(
        reference_without_whitespace,
        prediction_without_whitespace,
    )
    return {
        "edit_distance": edit_distance,
        "reference_characters": len(reference_text),
        "cer": safe_ratio(edit_distance, len(reference_text)),
        "exact_match": predicted_text == reference_text,
        "whitespace_normalized_edit_distance": whitespace_edit_distance,
        "whitespace_normalized_reference_characters": len(
            reference_without_whitespace
        ),
        "whitespace_normalized_cer": safe_ratio(
            whitespace_edit_distance,
            len(reference_without_whitespace),
        ),
        "whitespace_normalized_exact_match": (
            prediction_without_whitespace == reference_without_whitespace
        ),
    }


class OCRValidationAccumulator:
    def __init__(self) -> None:
        self.pages = 0
        self.character_edits = 0
        self.reference_characters = 0
        self.exact_matches = 0
        self.whitespace_edits = 0
        self.whitespace_reference_characters = 0
        self.whitespace_exact_matches = 0

    def add_page(self, reference_text: str, predicted_text: str) -> dict[str, Any]:
        page = evaluate_ocr_page(reference_text, predicted_text)
        self.pages += 1
        self.character_edits += int(page["edit_distance"])
        self.reference_characters += int(page["reference_characters"])
        self.exact_matches += int(bool(page["exact_match"]))
        self.whitespace_edits += int(page["whitespace_normalized_edit_distance"])
        self.whitespace_reference_characters += int(
            page["whitespace_normalized_reference_characters"]
        )
        self.whitespace_exact_matches += int(
            bool(page["whitespace_normalized_exact_match"])
        )
        return page

    def summary(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "character_edits": self.character_edits,
            "reference_characters": self.reference_characters,
            "page_cer": safe_ratio(
                self.character_edits,
                self.reference_characters,
            ),
            "mean_page_edit_distance": safe_ratio(
                self.character_edits,
                self.pages,
            ),
            "page_exact_matches": self.exact_matches,
            "page_exact_match_rate": safe_ratio(self.exact_matches, self.pages),
            "whitespace_normalized_character_edits": self.whitespace_edits,
            "whitespace_normalized_reference_characters": (
                self.whitespace_reference_characters
            ),
            "whitespace_normalized_page_cer": safe_ratio(
                self.whitespace_edits,
                self.whitespace_reference_characters,
            ),
            "whitespace_normalized_page_exact_matches": (
                self.whitespace_exact_matches
            ),
            "whitespace_normalized_page_exact_match_rate": safe_ratio(
                self.whitespace_exact_matches,
                self.pages,
            ),
        }


def _validated_box(value: Sequence[float], context: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{context} must contain four coordinates.")
    box = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in box):
        raise ValueError(f"{context} contains a non-finite coordinate.")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"{context} must be xyxy with positive area, got {box}.")
    return box


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    first_box = _validated_box(first, "first box")
    second_box = _validated_box(second, "second box")
    intersection_width = max(
        0.0,
        min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]),
    )
    intersection_height = max(
        0.0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class _FlowEdge:
    target: int
    reverse: int
    capacity: int
    cost: float


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: float,
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    backward = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def match_regions(
    predicted_boxes: Sequence[Sequence[float]],
    target_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> list[dict[str, int | float]]:
    """Maximum-cardinality, maximum-IoU matching above a fixed IoU threshold."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")
    predicted = [
        _validated_box(box, f"predicted_boxes[{index}]")
        for index, box in enumerate(predicted_boxes)
    ]
    targets = [
        _validated_box(box, f"target_boxes[{index}]")
        for index, box in enumerate(target_boxes)
    ]
    if not predicted or not targets:
        return []

    source = 0
    predicted_offset = 1
    target_offset = predicted_offset + len(predicted)
    sink = target_offset + len(targets)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    pair_edges: dict[tuple[int, int], tuple[_FlowEdge, float]] = {}

    for predicted_index in range(len(predicted)):
        _add_flow_edge(graph, source, predicted_offset + predicted_index, 1, 0.0)
    for target_index in range(len(targets)):
        _add_flow_edge(graph, target_offset + target_index, sink, 1, 0.0)
    for predicted_index, predicted_box in enumerate(predicted):
        for target_index, target_box in enumerate(targets):
            iou = box_iou(predicted_box, target_box)
            if iou + 1e-12 < iou_threshold:
                continue
            edge = _add_flow_edge(
                graph,
                predicted_offset + predicted_index,
                target_offset + target_index,
                1,
                -iou,
            )
            pair_edges[(predicted_index, target_index)] = (edge, iou)

    node_count = len(graph)
    while True:
        distances = [math.inf] * node_count
        previous_nodes = [-1] * node_count
        previous_edges = [-1] * node_count
        distances[source] = 0.0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not math.isfinite(distances[node]):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity < 1:
                        continue
                    candidate = distances[node] + edge.cost
                    if candidate < distances[edge.target] - 1e-12:
                        distances[edge.target] = candidate
                        previous_nodes[edge.target] = node
                        previous_edges[edge.target] = edge_index
                        changed = True
            if not changed:
                break
        if not math.isfinite(distances[sink]):
            break
        node = sink
        while node != source:
            previous_node = previous_nodes[node]
            if previous_node < 0:
                raise RuntimeError("Region matching produced an incomplete augmenting path.")
            edge = graph[previous_node][previous_edges[node]]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous_node

    matches = [
        {
            "prediction_index": predicted_index,
            "target_index": target_index,
            "iou": iou,
        }
        for (predicted_index, target_index), (edge, iou) in pair_edges.items()
        if edge.capacity == 0
    ]
    return sorted(matches, key=lambda match: int(match["prediction_index"]))


def _validate_regions(regions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise TypeError(f"regions[{index}] must be an object.")
        order = region.get("reading_order")
        if not isinstance(order, int):
            raise ValueError(f"regions[{index}].reading_order must be an integer.")
        direction = region.get("writing_direction", "unknown")
        if direction not in DIRECTION_LABELS:
            raise ValueError(f"regions[{index}] has unsupported direction {direction!r}.")
        validated.append(
            {
                **region,
                "bbox": _validated_box(region.get("bbox", ()), f"regions[{index}].bbox"),
                "reading_order": order,
                "writing_direction": direction,
            }
        )
    validated.sort(key=lambda region: int(region["reading_order"]))
    if [region["reading_order"] for region in validated] != list(range(len(validated))):
        raise ValueError("Region reading_order values must be contiguous from zero.")
    return validated


def evaluate_page(
    *,
    reference_text: str,
    predicted_text: str,
    regions: Sequence[dict[str, Any]],
    annotation_status: str,
    object_scores: Sequence[float],
    predicted_boxes: Sequence[Sequence[float]],
    predicted_directions: Sequence[int],
    object_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    if annotation_status not in {"complete", "partial", "none"}:
        raise ValueError(f"Unsupported layout annotation status: {annotation_status!r}.")
    if not 0.0 <= object_threshold <= 1.0:
        raise ValueError("object_threshold must be in [0, 1].")
    if not (
        len(object_scores) == len(predicted_boxes) == len(predicted_directions)
    ):
        raise ValueError("Object, box, and direction prediction counts must match.")
    scores = [float(score) for score in object_scores]
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("Object scores must be finite probabilities in [0, 1].")
    boxes = [
        _validated_box(box, f"predicted_boxes[{index}]")
        for index, box in enumerate(predicted_boxes)
    ]
    directions = [int(direction) for direction in predicted_directions]
    if any(direction < 0 or direction >= len(DIRECTION_LABELS) for direction in directions):
        raise ValueError("Predicted direction indices are out of range.")
    validated_regions = _validate_regions(regions)
    if annotation_status == "none" and validated_regions:
        raise ValueError("annotation_status='none' cannot contain regions.")
    if annotation_status == "complete" and not validated_regions:
        raise ValueError("annotation_status='complete' requires regions.")

    ocr_metrics = evaluate_ocr_page(reference_text, predicted_text)

    positive_indices = [
        index for index, score in enumerate(scores) if score >= object_threshold
    ]
    positive_boxes = [boxes[index] for index in positive_indices]
    target_boxes = [region["bbox"] for region in validated_regions]
    relative_matches = match_regions(positive_boxes, target_boxes, iou_threshold)
    matches = [
        {
            **match,
            "prediction_index": positive_indices[int(match["prediction_index"])],
        }
        for match in relative_matches
    ]

    ordered_ious = [
        box_iou(boxes[index], region["bbox"]) if index < len(boxes) else 0.0
        for index, region in enumerate(validated_regions)
    ]
    ordered_object_hits = sum(
        index < len(scores) and scores[index] >= object_threshold
        for index in range(len(validated_regions))
    )
    ordered_direction_hits = sum(
        index < len(directions)
        and DIRECTION_LABELS[directions[index]] == region["writing_direction"]
        for index, region in enumerate(validated_regions)
    )
    matched_direction_hits = sum(
        DIRECTION_LABELS[directions[int(match["prediction_index"])]]
        == validated_regions[int(match["target_index"])]["writing_direction"]
        for match in matches
    )

    concordant_pairs = 0
    discordant_pairs = 0
    for first_index, first_match in enumerate(matches):
        for second_match in matches[first_index + 1 :]:
            first_order = validated_regions[int(first_match["target_index"])]["reading_order"]
            second_order = validated_regions[int(second_match["target_index"])][
                "reading_order"
            ]
            if first_order < second_order:
                concordant_pairs += 1
            else:
                discordant_pairs += 1

    pair_count = concordant_pairs + discordant_pairs
    matched_iou_sum = sum(float(match["iou"]) for match in matches)
    page: dict[str, Any] = {
        "ocr": {
            **ocr_metrics,
        },
        "layout": {
            "annotation_status": annotation_status,
            "ground_truth_regions": len(validated_regions),
            "predicted_regions": len(positive_indices),
            "matched_regions": len(matches),
            "region_recall": safe_ratio(len(matches), len(validated_regions)),
            "region_precision": (
                safe_ratio(len(matches), len(positive_indices))
                if annotation_status == "complete"
                else None
            ),
            "ordered_slot_object_recall": safe_ratio(
                ordered_object_hits,
                len(validated_regions),
            ),
            "ordered_slot_bbox_mean_iou": safe_ratio(
                sum(ordered_ious),
                len(ordered_ious),
            ),
            "matched_bbox_mean_iou": safe_ratio(matched_iou_sum, len(matches)),
            "ordered_direction_accuracy": safe_ratio(
                ordered_direction_hits,
                len(validated_regions),
            ),
            "matched_direction_accuracy": safe_ratio(
                matched_direction_hits,
                len(matches),
            ),
            "reading_order_pair_accuracy": safe_ratio(concordant_pairs, pair_count),
            "reading_order_kendall_tau": safe_ratio(
                concordant_pairs - discordant_pairs,
                pair_count,
            ),
            "reading_order_pairs": pair_count,
            "matches": matches,
        },
        "counts": {
            "ordered_object_hits": ordered_object_hits,
            "ordered_bbox_iou_sum": sum(ordered_ious),
            "ordered_direction_hits": ordered_direction_hits,
            "matched_bbox_iou_sum": matched_iou_sum,
            "matched_direction_hits": matched_direction_hits,
            "concordant_reading_order_pairs": concordant_pairs,
            "discordant_reading_order_pairs": discordant_pairs,
        },
    }
    return page


class LayoutValidationAccumulator:
    def __init__(self, object_threshold: float = 0.5, iou_threshold: float = 0.5) -> None:
        if not 0.0 <= object_threshold <= 1.0:
            raise ValueError("object_threshold must be in [0, 1].")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        self.object_threshold = float(object_threshold)
        self.iou_threshold = float(iou_threshold)
        self.pages = 0
        self.ocr_edits = 0
        self.ocr_reference_characters = 0
        self.ocr_exact = 0
        self.whitespace_edits = 0
        self.whitespace_reference_characters = 0
        self.whitespace_exact = 0
        self.annotated_pages = 0
        self.annotated_regions = 0
        self.matched_annotated_regions = 0
        self.complete_pages = 0
        self.complete_regions = 0
        self.complete_predictions = 0
        self.complete_matches = 0
        self.ordered_object_hits = 0
        self.ordered_bbox_iou_sum = 0.0
        self.ordered_direction_hits = 0
        self.matched_bbox_iou_sum = 0.0
        self.matched_direction_hits = 0
        self.reading_order_concordant = 0
        self.reading_order_discordant = 0
        self.reading_order_pages_evaluable = 0

    def add_page(self, **kwargs: Any) -> dict[str, Any]:
        page = evaluate_page(
            object_threshold=self.object_threshold,
            iou_threshold=self.iou_threshold,
            **kwargs,
        )
        ocr = page["ocr"]
        layout = page["layout"]
        counts = page["counts"]
        self.pages += 1
        self.ocr_edits += int(ocr["edit_distance"])
        self.ocr_reference_characters += int(ocr["reference_characters"])
        self.ocr_exact += int(bool(ocr["exact_match"]))
        self.whitespace_edits += int(ocr["whitespace_normalized_edit_distance"])
        self.whitespace_reference_characters += int(
            ocr["whitespace_normalized_reference_characters"]
        )
        self.whitespace_exact += int(bool(ocr["whitespace_normalized_exact_match"]))

        ground_truth_regions = int(layout["ground_truth_regions"])
        matched_regions = int(layout["matched_regions"])
        if ground_truth_regions:
            self.annotated_pages += 1
            self.annotated_regions += ground_truth_regions
            self.matched_annotated_regions += matched_regions
        if layout["annotation_status"] == "complete":
            self.complete_pages += 1
            self.complete_regions += ground_truth_regions
            self.complete_predictions += int(layout["predicted_regions"])
            self.complete_matches += matched_regions
        self.ordered_object_hits += int(counts["ordered_object_hits"])
        self.ordered_bbox_iou_sum += float(counts["ordered_bbox_iou_sum"])
        self.ordered_direction_hits += int(counts["ordered_direction_hits"])
        self.matched_bbox_iou_sum += float(counts["matched_bbox_iou_sum"])
        self.matched_direction_hits += int(counts["matched_direction_hits"])
        self.reading_order_concordant += int(counts["concordant_reading_order_pairs"])
        self.reading_order_discordant += int(counts["discordant_reading_order_pairs"])
        if int(layout["reading_order_pairs"]) > 0:
            self.reading_order_pages_evaluable += 1
        return page

    def summary(self) -> dict[str, Any]:
        complete_precision = safe_ratio(self.complete_matches, self.complete_predictions)
        complete_recall = safe_ratio(self.complete_matches, self.complete_regions)
        complete_f1 = None
        if complete_precision is not None and complete_recall is not None:
            denominator = complete_precision + complete_recall
            complete_f1 = (
                2.0 * complete_precision * complete_recall / denominator
                if denominator > 0.0
                else 0.0
            )
        reading_pairs = self.reading_order_concordant + self.reading_order_discordant
        return {
            "thresholds": {
                "object_probability": self.object_threshold,
                "region_iou": self.iou_threshold,
            },
            "ocr": {
                "pages": self.pages,
                "character_edits": self.ocr_edits,
                "reference_characters": self.ocr_reference_characters,
                "page_cer": safe_ratio(self.ocr_edits, self.ocr_reference_characters),
                "mean_page_edit_distance": safe_ratio(self.ocr_edits, self.pages),
                "page_exact_matches": self.ocr_exact,
                "page_exact_match_rate": safe_ratio(self.ocr_exact, self.pages),
                "whitespace_normalized_character_edits": self.whitespace_edits,
                "whitespace_normalized_reference_characters": (
                    self.whitespace_reference_characters
                ),
                "whitespace_normalized_page_cer": safe_ratio(
                    self.whitespace_edits,
                    self.whitespace_reference_characters,
                ),
                "whitespace_normalized_page_exact_matches": self.whitespace_exact,
                "whitespace_normalized_page_exact_match_rate": safe_ratio(
                    self.whitespace_exact,
                    self.pages,
                ),
            },
            "layout": {
                "annotated_pages": self.annotated_pages,
                "annotated_regions": self.annotated_regions,
                "matched_annotated_regions": self.matched_annotated_regions,
                "region_recall": safe_ratio(
                    self.matched_annotated_regions,
                    self.annotated_regions,
                ),
                "complete_pages": self.complete_pages,
                "complete_ground_truth_regions": self.complete_regions,
                "complete_predicted_regions": self.complete_predictions,
                "complete_matched_regions": self.complete_matches,
                "complete_region_precision": complete_precision,
                "complete_region_recall": complete_recall,
                "complete_region_f1": complete_f1,
                "ordered_slot_object_recall": safe_ratio(
                    self.ordered_object_hits,
                    self.annotated_regions,
                ),
                "ordered_slot_bbox_mean_iou": safe_ratio(
                    self.ordered_bbox_iou_sum,
                    self.annotated_regions,
                ),
                "matched_bbox_mean_iou": safe_ratio(
                    self.matched_bbox_iou_sum,
                    self.matched_annotated_regions,
                ),
                "ordered_direction_accuracy": safe_ratio(
                    self.ordered_direction_hits,
                    self.annotated_regions,
                ),
                "matched_direction_accuracy": safe_ratio(
                    self.matched_direction_hits,
                    self.matched_annotated_regions,
                ),
                "reading_order_pairs": reading_pairs,
                "reading_order_pages_evaluable": self.reading_order_pages_evaluable,
                "reading_order_pair_accuracy": safe_ratio(
                    self.reading_order_concordant,
                    reading_pairs,
                ),
                "reading_order_kendall_tau": safe_ratio(
                    self.reading_order_concordant - self.reading_order_discordant,
                    reading_pairs,
                ),
            },
        }
