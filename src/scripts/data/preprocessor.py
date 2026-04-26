import abc


def validate_shard_alignment(shard_data: dict, num_samples: int) -> None:
    """Assert every per-sample field in `shard_data` has first-dim length
    equal to `num_samples`.

    Per-sample fields are: tensors with `ndim >= 1` (compared via `shape[0]`)
    and lists/tuples (compared via `len()`). Scalar values, dicts, and the
    `num_samples` key itself are ignored as metadata.

    Raises RuntimeError listing every mismatched field. Intended to be called
    right before `torch.save` in each preprocessor's flush_shard so an
    accumulator-lifecycle bug fails loudly on the first shard write rather
    than after the dataset is on disk.
    """
    mismatches = []
    for k, v in shard_data.items():
        if k == "num_samples":
            continue
        if hasattr(v, "shape") and hasattr(v, "ndim") and v.ndim >= 1:
            if v.shape[0] != num_samples:
                mismatches.append(
                    f"  {k}: shape[0]={v.shape[0]} (expected {num_samples})"
                )
        elif isinstance(v, (list, tuple)):
            if len(v) != num_samples:
                mismatches.append(
                    f"  {k}: len={len(v)} (expected {num_samples})"
                )
    if mismatches:
        raise RuntimeError(
            "Shard alignment check failed — per-sample fields don't all match "
            f"num_samples={num_samples}. This usually means an accumulator-"
            "lifecycle bug (populate gate and reset gate out of sync). "
            "Affected fields:\n" + "\n".join(mismatches)
        )


class BatchProcessor(abc.ABC):
    @abc.abstractmethod
    def process_batch(self, batch):
        pass


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def flush_shard(self):
        pass

    @abc.abstractmethod
    def process_and_accumulate(self):
        pass

    @abc.abstractmethod
    def preprocess_example(self, example) -> bool:
        pass

    @abc.abstractmethod
    def parse_config(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def add_cli_args(cls, parser):
        pass
