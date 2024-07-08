import os
import random
import datetime
from multiprocessing import Process
from torchvision import datasets
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import ImageFolder
from webdataset import TarWriter


def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size = 0
    for item in samples:
        size += stream.write(map_func(item))
    stream.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size


if __name__ == "__main__":
    root = "./"
    items = []
    # e.g., [('1.jpg', 'text_1'), ('2.jpg', 'text_2'), ('3.jpg', 'text_3'), ('4.jpg', 'text_4')]

    def map_func(item):
        img_path, txt = item
        with open(os.path.join(root+img_path), "rb") as stream:
            image = stream.read()
        sample = {
            "__key__": img_path,
            "img": image,
            "txt": txt
        }
        return sample

    make_wds_shards(
        pattern="./tars/test-%06d.tar",
        num_shards=2,
        num_workers=2,
        samples=items,
        map_func=map_func,
    )
