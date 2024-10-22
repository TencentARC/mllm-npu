import os
import json
import tarfile
import warnings
from io import BufferedIOBase
from typing import (cast, IO, Iterator, Optional, Tuple, Dict)

import hydra
import torchdata.datapipes as dp
import torch.distributed as dist
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("load_from_tar_wo_exception")
class TarArchiveLoaderWoException(TarArchiveLoader):

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(
                        data_stream.file_obj, tarfile.TarFile):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (self.mode
                                    if hasattr(data_stream, "seekable")
                                    and data_stream.seekable() else
                                    self.mode.replace(":", "|"))

                    tar = tarfile.open(fileobj=cast(Optional[IO[bytes]],
                                                    data_stream),
                                       mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(
                            f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                        )
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(
                        os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj,
                                                        data_stream,
                                                        name=inner_pathname)
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
                )
                # raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()


@functional_datapipe("parse_jsonl_files")
class JsonlParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]],
                 **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for file_name, stream in self.source_datapipe:
            for idx, line in enumerate(stream):
                if line.strip() != '':
                    try:
                        yield f'{file_name}_line{idx}', json.loads(line)
                    except Exception as e:
                        warnings.warn(
                            f"Error occured when parsing string to json due to: {e} abort!"
                        )


def build_multi_datapipes(datapipes,
                          tokenizer=None,
                          image_transform=None,
                          sample_weights=None):
    """
    load dataset settings
    """
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [
        hydra.utils.instantiate(datapipe,
                                tokenizer=tokenizer,
                                image_transform=image_transform)
        for datapipe in datapipes
    ]

    datasets_to_weights_dict = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_to_weights_dict[dataset] = sample_weight
    datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict,
                                         seed=888 + dist.get_rank())

    return datapipe
