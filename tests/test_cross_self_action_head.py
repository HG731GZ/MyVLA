import torch
from types import SimpleNamespace

from models.modeling_smolvlm_vla import _pack_vlm_inputs
from models.transformer_smolvlm import (
    CrossAttention,
    SmolVLMCrossSelfActionTransformer,
)
from train_smolvlm import build_optimizer, update_group_lrs


def _small_head(dropout: float = 0.0) -> SmolVLMCrossSelfActionTransformer:
    return SmolVLMCrossSelfActionTransformer(
        hidden_size=32,
        vlm_hidden_size=24,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        dim_action=7,
        dim_propio=8,
        dim_time=8,
        max_num_actions=10,
        dropout=dropout,
    )


def test_cross_self_head_shape_and_all_cross_attention_gradients():
    torch.manual_seed(0)
    model = _small_head()
    output = model(
        vlm_features=torch.randn(2, 6, 24),
        vlm_attention_mask=torch.tensor(
            [[True, True, True, True, False, False], [True] * 6]
        ),
        action_with_noise=torch.randn(2, 10, 7),
        proprio=torch.randn(2, 8),
        t=torch.rand(2),
    )

    assert output.shape == (2, 10, 7)
    output.square().mean().backward()
    for block in model.blocks:
        assert block.cross_attn.q_proj.weight.grad is not None
        assert torch.count_nonzero(block.cross_attn.q_proj.weight.grad) > 0


def test_masked_vlm_tokens_do_not_change_action_output():
    torch.manual_seed(1)
    model = _small_head().eval()
    vlm_features = torch.randn(2, 6, 24)
    vlm_mask = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, False, False]]
    )
    changed_features = vlm_features.clone()
    changed_features[~vlm_mask] = torch.randn_like(changed_features[~vlm_mask]) * 100
    action = torch.randn(2, 10, 7)
    proprio = torch.randn(2, 8)
    t = torch.rand(2)

    expected = model(vlm_features, action, proprio, t, vlm_mask)
    actual = model(changed_features, action, proprio, t, vlm_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cross_attention_sdpa_and_fallback_match():
    torch.manual_seed(2)
    attention = CrossAttention(dim=32, num_heads=4).eval()
    query = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    mask = torch.tensor(
        [[True, True, False, True, False, False, False], [True] * 7]
    )

    attention.fused_attn = True
    fused = attention(query, memory, mask)
    attention.fused_attn = False
    fallback = attention(query, memory, mask)

    torch.testing.assert_close(fallback, fused, rtol=1e-5, atol=1e-6)


def test_non_contiguous_image_mask_keeps_later_slot_and_text_mask():
    image_features = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(
        1,
        4,
        2,
        3,
    )
    image_mask = torch.tensor([[True, True, False, True]])
    text_embeds = torch.tensor([[[100.0] * 3, [200.0] * 3, [300.0] * 3]])
    text_mask = torch.tensor([[True, False, True]])

    packed, packed_mask = _pack_vlm_inputs(
        image_features,
        image_mask,
        text_embeds,
        text_mask,
    )

    expected = torch.cat(
        [
            image_features[0, 0],
            image_features[0, 1],
            image_features[0, 3],
            text_embeds[0, [0, 2]],
        ],
        dim=0,
    )
    torch.testing.assert_close(packed[0, packed_mask[0]], expected)


def test_optimizer_trains_full_action_head_while_vlm_is_frozen():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = torch.nn.Linear(3, 4)
            self.transformer = torch.nn.Linear(4, 2)

    model = DummyModel()
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.0)
    args = SimpleNamespace(
        learning_rate=1e-4,
        learning_coef=0.1,
        freeze_vlm_steps=1000,
        warmup_steps=0,
        iters=10000,
        min_lr_ratio=0.1,
        use_cosine_decay=False,
    )

    update_group_lrs(optimizer, 0, args)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs == {"vlm": 0.0, "action_head": 1e-4}

    update_group_lrs(optimizer, 1000, args)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs == {"vlm": 1e-5, "action_head": 1e-4}
