import hydra
from torchdata.dataloader2 import (DataLoader2, MultiProcessingReadingService,
                                   DistributedReadingService,
                                   SequentialReadingService)


def build_dataloader(dataset_cfg, image_transform, tokenizer,
                     dataloader_num_workers):
    dataset = hydra.utils.instantiate(dataset_cfg,
                                      image_transform=image_transform,
                                      tokenizer=tokenizer)
    mp_service = MultiProcessingReadingService(
        num_workers=dataloader_num_workers)
    dist_service = DistributedReadingService()
    reading_service = SequentialReadingService(dist_service, mp_service)
    dataloader = DataLoader2(dataset, reading_service=reading_service)

    return dataloader
