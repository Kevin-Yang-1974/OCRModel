import torch

from layout_ocr.geometry import box_giou_loss, box_iou, generalized_box_iou


def test_geometry_metrics_cover_overlap_cases() -> None:
    boxes_a = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]]
    )
    boxes_b = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 3.0, 2.0]]]
    )

    iou = box_iou(boxes_a, boxes_b)
    giou = generalized_box_iou(boxes_a, boxes_b)

    torch.testing.assert_close(iou[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(giou[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(iou[0, 1, 1], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(giou[0, 1, 1], torch.tensor(1.0 / 3.0))


def test_generalized_iou_reaches_negative_one_for_disjoint_degenerate_boxes() -> None:
    boxes_a = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    boxes_b = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

    torch.testing.assert_close(
        generalized_box_iou(boxes_a, boxes_b), torch.tensor([[[-1.0]]])
    )


def test_box_giou_loss_returns_per_query_values() -> None:
    pred = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 3.0, 2.0]]]
    )
    target = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]]
    )

    loss = box_giou_loss(pred, target)

    torch.testing.assert_close(loss, torch.tensor([[0.0, 2.0 / 3.0]]))
