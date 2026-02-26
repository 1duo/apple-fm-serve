from apple_fm_serve.stream import text_deltas_from_snapshots


def test_text_deltas_from_snapshots() -> None:
    snapshots = ["H", "He", "Hello"]
    assert list(text_deltas_from_snapshots(snapshots)) == ["H", "e", "llo"]


def test_text_deltas_recover_on_divergence() -> None:
    snapshots = ["abc", "ab", "xyz"]
    assert list(text_deltas_from_snapshots(snapshots)) == ["abc", "ab", "xyz"]
