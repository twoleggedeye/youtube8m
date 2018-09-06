import torch.multiprocessing as multiprocessing


class _DataLoaderWorker(multiprocessing.Process):
    def __init__(self, base_data_loader, batch_queue):
        super().__init__()
        self._base_data_loader = base_data_loader
        self._batch_queue = batch_queue

    def run(self):
        while True:
            for input_batch, output_batch in self._base_data_loader:
                self._batch_queue.put((input_batch, output_batch))


class ParallelDataLoader:
    def __init__(self, base_data_loader, epoch_size, process_count, max_queue_size):
        self._base_data_loader = base_data_loader
        self._epoch_size = epoch_size
        self._process_count = process_count
        self._max_queue_size = max_queue_size

    def __iter__(self):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        batch_queue = multiprocessing.Queue(maxsize=self._max_queue_size)
        workers = []
        try:
            for _ in range(self._process_count):
                worker = _DataLoaderWorker(self._base_data_loader, batch_queue)
                worker.start()
                workers.append(worker)

            yielded_size = 0

            while yielded_size < self._epoch_size:
                input_batch, output_batch = batch_queue.get()
                yielded_size += output_batch.shape[0]
                yield input_batch, output_batch

        finally:
            for worker in workers:
                try:
                    worker.terminate()
                except Exception:
                    pass
