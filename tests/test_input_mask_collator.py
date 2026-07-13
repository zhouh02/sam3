import functools
import json

import torch
from PIL import Image as PILImage

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
    Sam3ImageDataset,
)
from sam3.train.transforms.basic import get_random_resize_max_size
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    PadToSizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.segmentation import DecodeRle


def _metadata(object_id=0):
    return InferenceMetadata(
        coco_image_id=1,
        original_image_id=1,
        original_category_id=1,
        original_size=(8, 8),
        object_id=object_id,
        frame_index=0,
    )


def _datapoint(query, *, with_box=False):
    bbox = (
        torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32) if with_box else None
    )
    bbox_label = torch.ones(1, dtype=torch.long) if with_box else None
    query.input_bbox = bbox
    query.input_bbox_label = bbox_label
    return Datapoint(
        images=[
            Image(
                data=torch.zeros(3, 8, 8),
                objects=[
                    Object(
                        bbox=torch.tensor([0.25, 0.25, 0.5, 0.5]),
                        area=16.0,
                        object_id=0,
                    )
                ],
                size=(8, 8),
            )
        ],
        find_queries=[query],
    )


def _query_without_mask(query_text="object", *, with_box=False, object_id=0):
    query = FindQueryLoaded(
        query_text=query_text,
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        inference_metadata=_metadata(object_id=object_id),
    )
    return _datapoint(query, with_box=with_box)


def test_collator_batches_all_empty_input_masks_without_breaking_boxes_or_targets():
    collated = collate_fn_api(
        [
            _query_without_mask("object", with_box=True, object_id=0),
            _query_without_mask("thing", with_box=True, object_id=1),
        ],
        dict_key="test",
    )
    batched = collated["test"]
    find_input = batched.find_inputs[0]
    find_target = batched.find_targets[0]

    assert find_input.input_masks.shape == torch.Size([0, 2, 1, 8, 8])
    assert find_input.input_masks_label.shape == torch.Size([0, 2])
    assert find_input.input_masks_mask.shape == torch.Size([2, 0])
    assert find_input.input_mask_object_ids.shape == torch.Size([0, 2])

    assert find_input.input_boxes.shape == torch.Size([1, 2, 4])
    assert find_input.input_boxes_label.shape == torch.Size([1, 2])
    assert find_input.input_boxes_mask.shape == torch.Size([2, 1])
    assert find_input.input_boxes_mask.tolist() == [[False], [False]]
    assert batched.find_text_batch == ["object", "thing"]
    assert find_target.num_boxes.tolist() == [1, 1]
    assert find_target.object_ids_padded.tolist() == [[0], [0]]


def test_collator_batches_input_mask_prompts_with_padding():
    input_mask = torch.zeros(8, 8, dtype=torch.uint8)
    input_mask[2:6, 2:6] = 1
    query_with_mask = FindQueryLoaded(
        query_text="object",
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        input_mask=input_mask,
        input_mask_label=1,
        input_mask_object_id=11,
        inference_metadata=_metadata(),
    )
    query_without_mask = FindQueryLoaded(
        query_text="object",
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        inference_metadata=_metadata(),
    )

    collated = collate_fn_api(
        [_datapoint(query_with_mask), _datapoint(query_without_mask)],
        dict_key="test",
    )
    find_input = collated["test"].find_inputs[0]

    assert find_input.input_masks.shape == torch.Size([1, 2, 1, 8, 8])
    assert find_input.input_masks_label.shape == torch.Size([1, 2])
    assert find_input.input_masks_mask.shape == torch.Size([2, 1])
    assert find_input.input_mask_object_ids.shape == torch.Size([1, 2])

    assert find_input.input_masks_mask[0, 0].item() is False
    assert find_input.input_masks_mask[1, 0].item() is True
    assert find_input.input_masks_label[0, 0].item() == 1
    assert find_input.input_masks_label[0, 1].item() == 0
    assert find_input.input_mask_object_ids[0, 0].item() == 11
    assert find_input.input_mask_object_ids[0, 1].item() == -1
    assert find_input.input_masks[0, 0, 0].sum().item() == 16
    assert find_input.input_masks[0, 1, 0].sum().item() == 0


def _write_minimal_coco(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_id in [1, 2]:
        PILImage.new("RGB", (8, 8), color=(127, 127, 127)).save(
            image_dir / f"{image_id:012d}.jpg"
        )

    annotations = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "height": 8, "width": 8},
            {"id": 2, "file_name": "000000000002.jpg", "height": 8, "width": 8},
        ],
        "categories": [
            {"id": 1, "name": "widget"},
            {"id": 2, "name": "gadget"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
                "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
            },
            {
                "id": 102,
                "image_id": 1,
                "category_id": 1,
                "bbox": [3, 3, 4, 4],
                "area": 16,
                "iscrowd": 0,
                "segmentation": [[3, 3, 7, 3, 7, 7, 3, 7]],
            },
            {
                "id": 201,
                "image_id": 2,
                "category_id": 2,
                "bbox": [1, 1, 3, 3],
                "area": 9,
                "iscrowd": 0,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
            },
            {
                "id": 202,
                "image_id": 2,
                "category_id": 2,
                "bbox": [4, 4, 3, 3],
                "area": 9,
                "iscrowd": 0,
                "segmentation": [[4, 4, 7, 4, 7, 7, 4, 7]],
            },
        ],
    }
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(annotations), encoding="utf-8")
    return image_dir, ann_file


def test_coco_dataset_transforms_and_collator_populate_findstage_input_masks(tmp_path):
    image_dir, ann_file = _write_minimal_coco(tmp_path)
    coco_json_loader = functools.partial(
        COCO_FROM_JSON,
        include_negatives=False,
        category_chunk_size=2,
        use_mask_prompt=True,
        min_instances_per_prompt_category=2,
        prompt_sampling_policy="median_area",
        fixed_val_prompt_policy="median_area",
    )
    transforms = [
        ComposeAPI(
            [
                DecodeRle(),
                RandomResizeAPI(
                    sizes=8,
                    max_size=get_random_resize_max_size(8),
                    square=True,
                    consistent_transform=False,
                ),
                PadToSizeAPI(size=10, consistent_transform=False, bottom_right=True),
                ToTensorAPI(),
            ]
        )
    ]
    dataset = Sam3ImageDataset(
        img_folder=str(image_dir),
        ann_file=str(ann_file),
        transforms=transforms,
        max_ann_per_img=100000,
        multiplier=1,
        training=False,
        load_segmentation=True,
        max_val_queries=100000,
        use_caching=False,
        coco_json_loader=coco_json_loader,
    )
    dataset.set_curr_epoch(0)

    datapoints = [dataset[0], dataset[1]]
    prompt_object_ids = []
    for datapoint in datapoints:
        query = datapoint.find_queries[0]
        assert query.input_mask is not None
        assert query.input_mask_object_id is not None
        assert query.input_mask_object_id in query.object_ids_output
        prompt_object_ids.append(query.input_mask_object_id)

    collated = collate_fn_api(datapoints, dict_key="test")
    batched = collated["test"]
    stage = batched.find_inputs[0]

    assert stage.input_masks.ndim == 5
    n_masks, batch_size, channels, mask_h, mask_w = stage.input_masks.shape
    assert n_masks == 1
    assert batch_size == 2
    assert channels == 1
    assert stage.input_masks_label.shape == (n_masks, batch_size)
    assert stage.input_masks_mask.shape == (batch_size, n_masks)

    _, _, image_h, image_w = batched.img_batch.shape
    assert (mask_h, mask_w) == (image_h, image_w)
    assert (mask_h, mask_w) == (10, 10)

    for batch_index in range(batch_size):
        assert stage.input_masks_mask[batch_index, 0].item() is False
        assert stage.input_masks[0, batch_index].sum().item() > 0
        assert stage.input_masks_label[0, batch_index].item() == 1
        assert (
            stage.input_mask_object_ids[0, batch_index].item()
            == prompt_object_ids[batch_index]
        )

    unique_values = set(torch.unique(stage.input_masks).tolist())
    assert unique_values <= {0.0, 1.0}
    assert stage.input_masks.dtype in (torch.uint8, torch.bool, torch.float32)
