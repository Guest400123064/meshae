import json
import logging
import shutil
import sys
from argparse import ArgumentParser, Namespace
from typing import Callable, Sequence

import torch
import tqdm
from pathlib import Path
from PIL import Image
from PIL.ImageFile import ImageFile
from transformers import ViTImageProcessor, ViTModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger("embed_images.py")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=r"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def make_embed_func(
    name: str
) -> Callable[[Sequence[ImageFile], bool], torch.Tensor]:
    r"""Load the processor-model and construct a image processor function."""

    processor = ViTImageProcessor.from_pretrained(name)
    model = ViTModel.from_pretrained(name).to(device)

    def func(images: Sequence[ImageFile], keep_cls: bool = True) -> torch.Tensor:
        r"""Embed images into patch embeddings.

        Parameters
        ----------
        images: Sequence[ImageFile]
            Sequence of Pillow ``ImageFile``.
        keep_cls: bool, default to ``True``
            Whether keep the embedding corresponding to the ``CLS`` token or not.

        Returns
        -------
        torch.Tensor
            Processed batch of images with shape ``(B, T, E)``. ``B`` should equal
            to ``len(images)``. ``E`` corresponds to embedding size, e.g., 768.
        """
        inputs = processor(images=images, return_tensors="pt").to(device)
        embeddings = model(**inputs).last_hidden_state

        if keep_cls:
            return embeddings
        return embeddings[:, 1:]

    logger.info(f"Successfully loaded pre-trained model from <{name}>!")
    return func


def get_args() -> Namespace:
    r"""Grab CLI arguments."""

    parser = ArgumentParser(
        prog="embed_images.py",
        description="Preprocess Pix3D images into patch embeddings.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the pre-trained ViT model",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to the Pix3D root directory.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help=(
            "Path to the patch embedding storage directory. Note that a sub-folder "
            "with the ViT model name will be created under this directory; slashes "
            "will be replaced by underscores to keep flat folder structure."
        )
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        required=False,
        default=32,
        help="Batch size for processing images to prevent memory issue.",
    )
    parser.add_argument(
        "--keep-cls",
        type=bool,
        required=False,
        default=True,
        help="Set to ``True`` to discard ``CLS`` embeddings."
    )
    args = parser.parse_args()
    args.output = args.output / args.model.replace("/", "_")

    return args


if __name__ == "__main__":
    args = get_args()
    func = make_embed_func(args.model)

    logger.info(f"Loading ``pix3d.json`` from <{args.input}>.")
    with open(args.input / "pix3d.json") as f:
        index = json.load(f)
        ptoi = {obj["img"]: i for i, obj in enumerate(index)}

    embeddings = []
    for i in tqdm.tqdm(
        range(0, len(ptoi.keys()), args.batch_size),
        desc="Embedding batches...",
    ):
        batch = [
            Image.open(args.input / obj["img"]).convert("RGB")
            for obj in index[i:(i + args.batch_size)]
        ]
        embeddings.append(func(batch, args.keep_cls).cpu())

    if args.output.exists():
        logger.info(f"Purging existing <{args.output}>...")
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True)
    torch.save(torch.cat(embeddings), args.output / "dump.pt")
    with open(args.output / "ptoi.json", "w") as f:
        json.dump(ptoi, f, indent=True)

    logger.info(f"Embeddings dumped to <{args.output}>.")
