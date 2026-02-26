from collections.abc import AsyncIterator, Iterable, Iterator


def text_deltas_from_snapshots(snapshots: Iterable[str]) -> Iterator[str]:
    previous = ""
    for snapshot in snapshots:
        delta = snapshot.removeprefix(previous)
        if not delta:
            delta = snapshot
        previous = snapshot
        if delta:
            yield delta


async def async_text_deltas_from_snapshots(snapshots: AsyncIterator[str]) -> AsyncIterator[str]:
    previous = ""
    async for snapshot in snapshots:
        delta = snapshot.removeprefix(previous)
        if not delta:
            delta = snapshot
        previous = snapshot
        if delta:
            yield delta
