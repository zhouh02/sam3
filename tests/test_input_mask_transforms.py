import copy

import numpy as np
import torch
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image, Object
from sam3.train.transforms.basic_for_api import crop, hflip, pad, resize
from sam3.train.transforms.segmentation import DecodeRle


def _make_datapoint(input_mask, *, with_object=True):
    obj_mask = torch.zeros((4, 5), dtype=torch.uint8)
    obj_mask[1:3, 1:4] = 1
    objects = []
    if with_object:
        objects.append(
            Object(
                bbox=torch.tensor([1.0, 1.0, 4.0, 3.0]),
                area=6.0,
                object_id=1,
                segment=obj_mask,
            )
        )

    return Datapoint(
        images=[
            Image(
                data=PILImage.new("RGB", (5, 4)),
                objects=objects,
                size=(4, 5),
            )
        ],
        find_queries=[
            FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[0] if with_object else [],
                is_exhaustive=True,
                input_mask=input_mask,
                semantic_target=obj_mask.clone() if with_object else None,
            )
        ],
    )


def _rle_from_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_decode_rle_decodes_query_input_mask_rle_to_single_channel_tensor():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    datapoint = _make_datapoint(_rle_from_mask(mask))

    datapoint = DecodeRle()(datapoint)

    input_mask = datapoint.find_queries[0].input_mask
    assert isinstance(input_mask, torch.Tensor)
    assert input_mask.shape == torch.Size([4, 5])
    assert input_mask.dtype == torch.uint8
    assert torch.equal(input_mask, torch.from_numpy(mask))


def test_decode_rle_decodes_query_input_mask_polygon_to_single_channel_tensor():
    polygon = [[1, 1, 4, 1, 4, 3, 1, 3]]
    datapoint = _make_datapoint(copy.deepcopy(polygon))

    datapoint = DecodeRle()(datapoint)

    expected = torch.from_numpy(
        mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygon, 4, 5)))
    ).to(torch.uint8)
    input_mask = datapoint.find_queries[0].input_mask
    assert isinstance(input_mask, torch.Tensor)
    assert input_mask.shape == torch.Size([4, 5])
    assert input_mask.dtype == torch.uint8
    assert torch.equal(input_mask, expected)


def test_input_mask_follows_crop_hflip_resize_and_pad():
    mask = torch.zeros((4, 5), dtype=torch.uint8)
    mask[1:3, 1:4] = 1
    datapoint = _make_datapoint(mask.clone())

    datapoint = crop(datapoint, 0, (1, 1, 2, 3), check_validity=False)
    assert datapoint.find_queries[0].input_mask.shape == torch.Size([2, 3])
    assert datapoint.find_queries[0].input_mask.sum().item() == 6

    datapoint = hflip(datapoint, 0)
    assert datapoint.find_queries[0].input_mask.shape == torch.Size([2, 3])
    assert datapoint.find_queries[0].input_mask.sum().item() == 6

    datapoint = resize(datapoint, 0, (6, 4))
    assert datapoint.find_queries[0].input_mask.shape == torch.Size([4, 6])
    assert datapoint.find_queries[0].input_mask.dtype == torch.uint8
    assert datapoint.find_queries[0].input_mask.sum().item() > 0

    datapoint = pad(datapoint, 0, (2, 1))
    assert datapoint.find_queries[0].input_mask.shape == torch.Size([5, 8])
    assert datapoint.find_queries[0].input_mask.sum().item() > 0


def test_input_mask_none_is_ignored_by_spatial_transforms():
    datapoint = _make_datapoint(None, with_object=False)

    datapoint = DecodeRle()(datapoint)
    datapoint = crop(datapoint, 0, (1, 1, 2, 3), check_validity=False)
    datapoint = hflip(datapoint, 0)
    datapoint = resize(datapoint, 0, (6, 4))
    datapoint = pad(datapoint, 0, (2, 1))

    assert datapoint.find_queries[0].input_mask is None
