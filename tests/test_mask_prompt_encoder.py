import types

import pytest
import torch
from torch import nn

from sam3.model.geometry_encoders import (
    MaskEncoder,
    MaskPromptEncoder,
    Prompt,
    SequenceGeometryEncoder,
)
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model_builder import _create_geometry_encoder


class _ConstantBranch(nn.Module):
    def __init__(self, value: float, d_model: int):
        super().__init__()
        self.value = value
        self.d_model = d_model

    def forward(self, masks, *args, **kwargs):
        n_masks, batch_size = masks.shape[:2]
        return torch.full(
            (n_masks, batch_size, self.d_model),
            self.value,
            device=masks.device,
        )


def _make_inputs():
    masks = torch.zeros(2, 2, 1, 8, 8)
    masks[0, 0, 0, 1:3, 2:5] = 1
    masks[1, 0, 0, 4:7, 1:4] = 1
    masks[0, 1, 0, 2:6, 3:7] = 1
    # masks[1, 1] is padding and may remain empty.
    labels = torch.tensor([[1, 0], [0, -1]])
    padding_mask = torch.tensor([[False, False], [False, True]])
    image_features = torch.randn(2, 8, 4, 4)
    position_inputs = torch.randn(2, 2, 11)
    return masks, labels, padding_mask, image_features, position_inputs


def test_mask_prompt_encoder_adds_four_branches_without_fusion_norm():
    encoder = MaskPromptEncoder(d_model=8, position_input_dim=11)
    encoder._encode_mask_visual = types.MethodType(
        lambda self, masks, img_feats: _ConstantBranch(1, 8)(masks), encoder
    )
    encoder._encode_mask_shape = types.MethodType(
        lambda self, masks, padding_mask: _ConstantBranch(2, 8)(masks), encoder
    )
    encoder.mask_pos_enc_project = _ConstantBranch(3, 8)
    encoder.mask_label_embed = nn.Embedding(2, 8)
    nn.init.constant_(encoder.mask_label_embed.weight, 4)

    masks, labels, padding_mask, image_features, position_inputs = _make_inputs()
    tokens = encoder(
        masks=masks,
        mask_labels=labels,
        mask_padding_mask=padding_mask,
        img_feats=image_features,
        position_inputs=position_inputs,
    )

    expected = torch.full((2, 2, 8), 10.0)
    expected[1, 1] = 0
    assert torch.equal(tokens, expected)
    assert not any(
        isinstance(module, nn.LayerNorm) for module in encoder.modules()
    )


def test_mask_prompt_encoder_rejects_empty_non_padding_mask():
    encoder = MaskPromptEncoder(d_model=8, position_input_dim=11)
    masks = torch.zeros(1, 1, 1, 8, 8)

    with pytest.raises(
        ValueError,
        match="A non-padding mask prompt contains no foreground pixels",
    ):
        encoder(
            masks=masks,
            mask_labels=torch.ones(1, 1, dtype=torch.long),
            mask_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
            img_feats=torch.randn(1, 8, 4, 4),
            position_inputs=torch.randn(1, 1, 11),
        )


def test_mask_prompt_encoder_sanitizes_padding_labels_and_validates_real_labels():
    encoder = MaskPromptEncoder(d_model=8, position_input_dim=11)
    masks = torch.zeros(1, 2, 1, 8, 8)
    masks[0, 0, 0, 1:3, 1:3] = 1
    padding_mask = torch.tensor([[False], [True]])
    common = dict(
        masks=masks,
        mask_padding_mask=padding_mask,
        img_feats=torch.randn(2, 8, 4, 4),
        position_inputs=torch.randn(1, 2, 11),
    )

    tokens = encoder(mask_labels=torch.tensor([[1, -1]]), **common)
    assert torch.isfinite(tokens).all()
    assert torch.equal(tokens[0, 1], torch.zeros(8))

    with pytest.raises(ValueError, match="must be either 0 or 1"):
        encoder(mask_labels=torch.tensor([[2, -1]]), **common)


def test_mask_prompt_encoder_preserves_single_pixel_shape_and_one_token_per_mask():
    encoder = MaskPromptEncoder(d_model=8, position_input_dim=11)
    masks = torch.zeros(2, 1, 1, 8, 8)
    masks[0, 0, 0, 3, 5] = 1
    masks[1, 0, 0, 1:7, 2:6] = 1
    padding_mask = torch.zeros(1, 2, dtype=torch.bool)

    crops = encoder._crop_resize_masks(masks, padding_mask)
    tokens = encoder(
        masks=masks,
        mask_labels=torch.ones(2, 1, dtype=torch.long),
        mask_padding_mask=padding_mask,
        img_feats=torch.randn(1, 8, 4, 4),
        position_inputs=torch.randn(2, 1, 11),
    )

    assert crops.shape == (2, 1, 32, 32)
    assert crops[0].sum() > 0
    assert tokens.shape == (2, 1, 8)
    assert torch.isfinite(tokens).all()


def _make_geometry_encoder(d_model=8):
    return SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=False,
        points_pos_enc=False,
        boxes_direct_project=True,
        boxes_pool=False,
        boxes_pos_enc=True,
        d_model=d_model,
        pos_enc=PositionEmbeddingSine(d_model),
        num_layers=0,
        layer=nn.Identity(),
        add_cls=True,
        add_post_encode_proj=True,
        mask_prompt_encoder=MaskPromptEncoder(
            d_model=d_model,
            position_input_dim=d_model + 3,
        ),
    )


def test_prompt_keeps_raw_mask_prompt_fields_separate_from_dense_masks():
    raw_masks = torch.ones(1, 2, 1, 8, 8)
    labels = torch.ones(1, 2, dtype=torch.long)
    padding_mask = torch.zeros(2, 1, dtype=torch.bool)
    prompt = Prompt(
        box_embeddings=torch.zeros(0, 2, 4),
        box_mask=torch.zeros(2, 0, dtype=torch.bool),
        mask_prompt_inputs=raw_masks,
        mask_prompt_labels=labels,
        mask_prompt_mask=padding_mask,
    )

    cloned = prompt.clone()
    assert prompt.mask_embeddings is None
    assert torch.equal(cloned.mask_prompt_inputs, raw_masks)
    assert torch.equal(cloned.mask_prompt_labels, labels)
    assert torch.equal(cloned.mask_prompt_mask, padding_mask)


def test_encode_mask_prompts_matches_box_contract_and_geometry_convention():
    encoder = _make_geometry_encoder()
    masks = torch.zeros(1, 1, 1, 8, 10)
    masks[0, 0, 0, 2:5, 3:7] = 1
    labels = torch.ones(1, 1, dtype=torch.long)
    padding_mask = torch.zeros(1, 1, dtype=torch.bool)
    img_feats = torch.randn(1, 8, 4, 5)

    mask_tokens, returned_mask = encoder._encode_mask_prompts(
        masks=masks,
        mask_padding_mask=padding_mask,
        mask_labels=labels,
        img_feats=img_feats,
    )
    box_tokens, box_mask = encoder._encode_boxes(
        boxes=torch.tensor([[[0.5, 0.4375, 0.4, 0.375]]]),
        boxes_mask=padding_mask,
        boxes_labels=labels,
        img_feats=img_feats,
    )

    stats = encoder._compute_mask_geometry(masks, padding_mask)
    assert torch.allclose(stats[0], torch.tensor([[0.5]]))
    assert torch.allclose(stats[1], torch.tensor([[0.4375]]))
    assert torch.allclose(stats[2], torch.tensor([[0.4]]))
    assert torch.allclose(stats[3], torch.tensor([[0.375]]))
    assert torch.allclose(stats[4], torch.tensor([[0.15]]))
    assert mask_tokens.shape == box_tokens.shape == (1, 1, 8)
    assert returned_mask.shape == box_mask.shape == (1, 1)


def test_valid_mask_replaces_box_but_all_padding_box_path_is_preserved(monkeypatch):
    encoder = _make_geometry_encoder()
    calls = {"boxes": 0, "masks": 0}
    original_boxes = encoder._encode_boxes
    original_masks = encoder._encode_mask_prompts

    def count_boxes(*args, **kwargs):
        calls["boxes"] += 1
        return original_boxes(*args, **kwargs)

    def count_masks(*args, **kwargs):
        calls["masks"] += 1
        return original_masks(*args, **kwargs)

    monkeypatch.setattr(encoder, "_encode_boxes", count_boxes)
    monkeypatch.setattr(encoder, "_encode_mask_prompts", count_masks)
    img_feats = [torch.randn(4, 1, 8)]
    img_sizes = [(2, 2)]
    all_padding_box_prompt = Prompt(
        box_embeddings=torch.zeros(1, 1, 4),
        box_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    encoder(all_padding_box_prompt, img_feats, img_sizes)
    assert calls == {"boxes": 1, "masks": 0}

    mask_prompt = Prompt(
        box_embeddings=torch.zeros(1, 1, 4),
        box_mask=torch.ones(1, 1, dtype=torch.bool),
        mask_prompt_inputs=torch.ones(1, 1, 1, 8, 8),
        mask_prompt_labels=torch.ones(1, 1, dtype=torch.long),
        mask_prompt_mask=torch.zeros(1, 1, dtype=torch.bool),
    )
    output, output_mask = encoder(mask_prompt, img_feats, img_sizes)
    assert calls == {"boxes": 1, "masks": 1}
    assert output.shape == (2, 1, 8)  # one mask token plus CLS
    assert output_mask.shape == (1, 2)


def test_valid_box_and_valid_mask_are_rejected_together():
    encoder = _make_geometry_encoder()
    prompt = Prompt(
        box_embeddings=torch.ones(1, 1, 4),
        box_mask=torch.zeros(1, 1, dtype=torch.bool),
        mask_prompt_inputs=torch.ones(1, 1, 1, 8, 8),
        mask_prompt_labels=torch.ones(1, 1, dtype=torch.long),
        mask_prompt_mask=torch.zeros(1, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        encoder(prompt, [torch.randn(4, 1, 8)], [(2, 2)])


def test_sam3_image_copies_find_stage_mask_prompt_into_prompt():
    find_input = types.SimpleNamespace(
        input_points_before_embed=None,
        input_boxes_before_embed=torch.zeros(1, 1, 4),
        input_boxes_mask=torch.ones(1, 1, dtype=torch.bool),
        input_boxes_label=torch.zeros(1, 1, dtype=torch.long),
        input_masks=torch.ones(1, 1, 1, 8, 8),
        input_masks_label=torch.ones(1, 1, dtype=torch.long),
        input_masks_mask=torch.zeros(1, 1, dtype=torch.bool),
    )

    prompt = Sam3Image._get_geo_prompt_from_find_input(None, find_input)

    assert prompt.mask_embeddings is None
    assert prompt.mask_prompt_inputs is find_input.input_masks
    assert prompt.mask_prompt_labels is find_input.input_masks_label
    assert prompt.mask_prompt_mask is find_input.input_masks_mask


def test_default_geometry_encoder_builds_single_token_mask_encoder():
    encoder = _create_geometry_encoder()

    assert isinstance(encoder.mask_prompt_encoder, MaskPromptEncoder)
    assert encoder.mask_prompt_encoder.d_model == 256
    assert encoder.mask_prompt_encoder.mask_pos_enc_project.in_features == 259


class _PassThroughGeometryLayer(nn.Module):
    def forward(self, tgt, **kwargs):
        return tgt


def test_mask_prompt_uses_shared_projection_norm_and_geometry_transformer():
    encoder = SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=False,
        points_pos_enc=False,
        boxes_direct_project=True,
        boxes_pool=False,
        boxes_pos_enc=True,
        d_model=8,
        pos_enc=PositionEmbeddingSine(8),
        num_layers=1,
        layer=_PassThroughGeometryLayer(),
        add_cls=True,
        add_post_encode_proj=True,
        mask_prompt_encoder=MaskPromptEncoder(8, 11),
    )
    calls = []
    hooks = [
        encoder.final_proj.register_forward_hook(lambda *args: calls.append("proj")),
        encoder.norm.register_forward_hook(lambda *args: calls.append("norm")),
        encoder.encode[0].register_forward_hook(
            lambda *args: calls.append("transformer")
        ),
        encoder.encode_norm.register_forward_hook(
            lambda *args: calls.append("encode_norm")
        ),
    ]
    prompt = Prompt(
        box_embeddings=torch.zeros(1, 1, 4),
        box_mask=torch.ones(1, 1, dtype=torch.bool),
        mask_prompt_inputs=torch.ones(1, 1, 1, 8, 8),
        mask_prompt_labels=torch.ones(1, 1, dtype=torch.long),
        mask_prompt_mask=torch.zeros(1, 1, dtype=torch.bool),
    )

    output, output_mask = encoder(
        prompt,
        [torch.randn(4, 1, 8)],
        [(2, 2)],
        [torch.randn(4, 1, 8)],
    )
    for hook in hooks:
        hook.remove()

    assert calls == ["proj", "norm", "transformer", "encode_norm"]
    assert output.shape == (2, 1, 8)
    assert output_mask.shape == (1, 2)


def test_dense_mask_encoder_keeps_spatial_token_semantics():
    encoder = _make_geometry_encoder()
    encoder.mask_encoder = MaskEncoder(nn.Identity(), nn.Identity())
    encoder.add_mask_label = False
    dense_tokens, dense_mask = encoder._encode_masks(
        masks=torch.ones(1, 2, 1, 3, 4),
        attn_mask=torch.zeros(2, 1, dtype=torch.bool),
        mask_labels=torch.ones(1, 2, dtype=torch.long),
        img_feats=None,
    )

    assert dense_tokens.shape == (12, 2, 1)
    assert dense_mask.shape == (2, 12)
