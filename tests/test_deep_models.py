import pytest

torch = pytest.importorskip("torch")

from fiber.classify.rules import LABELS
from fiber.models.column_graph import ColumnGraph
from fiber.models.column_transformer import ColumnTransformer
from fiber.models.depth_cnn import DepthCNN
from fiber.models.patch_classifier import PatchClassifier


def test_depth_cnn_forward_uses_a_deeper_stack_than_the_patch_net():
    torch.manual_seed(0)
    model = DepthCNN()
    patch = PatchClassifier()
    batch = torch.randn(2, 1, 28, 6)
    out = model(batch)
    out.sum().backward()
    convs = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    patch_convs = [layer for layer in patch.modules() if isinstance(layer, torch.nn.Conv2d)]
    assert out.shape == (2, len(LABELS))
    assert len(convs) >= 2
    assert len(convs) > len(patch_convs)
    assert set(model.state_dict()) != set(patch.state_dict())
    assert model.stem.weight.grad is not None and model.stem.weight.grad.abs().sum() > 0
    assert model.block2.conv2.weight.grad is not None and model.block2.conv2.weight.grad.abs().sum() > 0


def test_column_graph_forward_passes_messages_between_neighbours():
    torch.manual_seed(0)
    model = ColumnGraph()
    batch = torch.randn(2, 1, 28, 6)
    out = model(batch)
    out.sum().backward()
    assert out.shape == (2, len(LABELS))
    assert isinstance(model.msg1, torch.nn.Linear)
    assert isinstance(model.msg2, torch.nn.Linear)
    assert isinstance(model.msg3, torch.nn.Linear)
    assert model.msg1.weight.grad is not None and model.msg1.weight.grad.abs().sum() > 0
    assert model.msg2.weight.grad is not None and model.msg2.weight.grad.abs().sum() > 0
    assert model.msg3.weight.grad is not None and model.msg3.weight.grad.abs().sum() > 0
    edge = torch.zeros(1, 1, 28, 6)
    middle = torch.zeros(1, 1, 28, 6)
    edge[0, 0, :, 0] = 1.0
    middle[0, 0, :, 3] = 1.0
    with torch.inference_mode():
        edge_out = model(edge)
        middle_out = model(middle)
    assert not torch.allclose(edge_out, middle_out)


def test_column_transformer_forward_stacks_two_attention_layers():
    torch.manual_seed(0)
    model = ColumnTransformer()
    batch = torch.randn(2, 1, 28, 6)
    out = model(batch)
    out.sum().backward()
    assert out.shape == (2, len(LABELS))
    assert len(model.encoder.layers) >= 2
    first = model.encoder.layers[0].self_attn
    second = model.encoder.layers[1].self_attn
    assert isinstance(first, torch.nn.MultiheadAttention)
    assert isinstance(second, torch.nn.MultiheadAttention)
    assert first.in_proj_weight.grad is not None and first.in_proj_weight.grad.abs().sum() > 0
    assert second.in_proj_weight.grad is not None and second.in_proj_weight.grad.abs().sum() > 0
