import abc


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
    def preprocess_example(self, example):
        pass

    @abc.abstractmethod
    def parse_config(self) -> dict:
        pass
