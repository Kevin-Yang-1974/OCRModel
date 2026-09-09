from layout_ocr.distributed import rank_epoch_indices


def test_rank_epoch_indices_covers_all_pages_without_dropping() -> None:
    record_count = 2159
    world_size = 5
    plans = [
        rank_epoch_indices(
            record_count,
            seed=42,
            epoch=7,
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    ]
    assert {len(plan) for plan in plans} == {432}
    flattened = [index for plan in plans for index in plan]
    assert len(flattened) == 2160
    assert set(flattened) == set(range(record_count))
    assert flattened[-1] == flattened[0]


def test_rank_epoch_indices_is_deterministic_and_epoch_specific() -> None:
    first = rank_epoch_indices(11, seed=42, epoch=0, rank=2, world_size=5)
    repeat = rank_epoch_indices(11, seed=42, epoch=0, rank=2, world_size=5)
    next_epoch = rank_epoch_indices(11, seed=42, epoch=1, rank=2, world_size=5)
    assert first == repeat
    assert first != next_epoch


def test_rank_epoch_indices_pads_complete_accumulation_batches() -> None:
    plans = [
        rank_epoch_indices(
            13,
            seed=42,
            epoch=0,
            rank=rank,
            world_size=5,
            batch_size=4,
        )
        for rank in range(5)
    ]
    assert {len(plan) for plan in plans} == {4}
    flattened = [index for plan in plans for index in plan]
    assert len(flattened) == 20
    assert set(flattened) == set(range(13))
