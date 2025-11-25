"""
结构化剪枝
"""
import os
import torch
import torch_pruning as tp
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from texo.data.processor import EvalMERImageProcessor
from texo.model.layer import LightConvBNAct


def _detect_device() -> str:
    """检测可用设备类型，优先使用 CUDA，其次 MPS，最后 CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_texo_model(model_dir: str = "./model"):
    """加载 Texo 的 VisionEncoderDecoder 模型与分词器"""
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def get_example_inputs(image_path: str | None = None):
    """生成一张示例输入；优先用真实图片，否则用随机张量"""
    try:
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            processor = EvalMERImageProcessor(image_size={"width": 384, "height": 384})
            pixel_values = processor(image).unsqueeze(0)
        else:
            pixel_values = torch.randn(1, 3, 384, 384)
    except Exception:
        pixel_values = torch.randn(1, 3, 384, 384)
    return pixel_values


def count_params(module: torch.nn.Module) -> int:
    """统计模块参数总数"""
    return sum(p.numel() for p in module.parameters())


def prune_encoder_channels(encoder: torch.nn.Module, example_inputs: torch.Tensor, pr_ratio: float = 0.2):
    """对编码器的 LightConvBNAct 的 pointwise 卷积进行结构化通道剪枝"""
    encoder.eval()
    DG = tp.DependencyGraph().build_dependency(encoder, example_inputs)
    for m in encoder.modules():
        if isinstance(m, LightConvBNAct):
            conv = m.conv1.conv
            if conv.groups != 1 or conv.out_channels <= 4:
                continue
            k = max(1, int(conv.out_channels * pr_ratio))
            w = conv.weight.detach().abs().sum(dim=(1, 2, 3))
            idxs = torch.topk(w, k, largest=False).indices.tolist()
            plan = DG.get_pruning_plan(conv, tp.prune_conv, idxs=idxs)
            plan.exec()
    return encoder


def main():
    """构建模型，执行编码器结构化剪枝，并打印剪枝前后参数变化"""
    device = _detect_device()
    model, tokenizer = build_texo_model()
    model.to(device)
    encoder = model.encoder

    example_inputs = get_example_inputs().to(device)
    before = count_params(encoder)
    prune_encoder_channels(encoder, example_inputs, pr_ratio=0.2)
    after = count_params(encoder)
    print(f"Encoder params: {before} -> {after} ({before/after:.2f}x)")

    try:
        torch.save(model.state_dict(), "model/torch_pruning_state.pt")
        print("Saved pruned state_dict to model/torch_pruning_state.pt")
    except Exception:
        pass


if __name__ == "__main__":
    main()