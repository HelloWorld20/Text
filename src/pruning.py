"""
非结构化剪枝
"""
from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet
from transformers import AutoTokenizer, VisionEncoderDecoderModel, PreTrainedTokenizerFast
from PIL import Image
import torch
from torch import nn

image_path = './TechnoSelection/test_img/单行公式.png'

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def inference(model: FormulaNet, image_path: str, tokenizer: PreTrainedTokenizerFast):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    image = Image.open(image_path)
    image_processor = EvalMERImageProcessor(image_size={'width':384, 'height':384})
    processed_image = image_processor(image).unsqueeze(0)
    outputs = model.generate(pixel_values=processed_image.to(device))
    pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return pred_str

def load(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    from pathlib import Path
    tok_file = Path(path).joinpath('tokenizer.json')
    if tok_file.exists():
        tokenizer = AutoTokenizer.from_pretrained(path, tokenizer_file=str(tok_file), use_fast=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    return model, tokenizer

def evaluate(model: FormulaNet, tokenizer: PreTrainedTokenizerFast):
    output = inference(model, image_path, tokenizer)

    pass

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance = tensor.abs()
    threshold = importance.view(-1).kthvalue(num_zeros).values
    mask = torch.gt(importance, threshold) # "gt" means "greater than"
    tensor.mul_(mask) # element-wise multiplication

    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                if isinstance(sparsity_dict, dict): # sparsity_dict is a dictionary
                    masks[name] = fine_grained_prune(param, sparsity_dict[name])
                else: # sparsity_dict can be a list
                    assert(sparsity_dict < 1 and sparsity_dict >= 0)
                    if sparsity_dict > 0:
                        masks[name] = fine_grained_prune(param, sparsity_dict)
        return masks


def main():
    model, tokenizer = load('./model')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # image_path = './TechnoSelection/test_img/单行公式.png'

    # print(model.named_parameters())

    dense_model_size = get_model_size(model)

    # set a sparsity of 0.99 for all layer types
    sparsity = 0.9

    pruner = FineGrainedPruner(model, sparsity)

    pruner.apply(model)

    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"{sparsity*100}% sparse model has size={sparse_model_size/MiB:.2f} MiB, "
        f"which is {dense_model_size/sparse_model_size:.2f}X smaller than "
        f"the {dense_model_size/MiB:.2f} MiB dense model")

    model.save_pretrained("model/pruning")

    # print(pruner.masks)

    # evaluate(model, tokenizer)


# display(IPython.display.Image(image_path))
# pred_str = inference(model, image_path, tokenizer, device)
# print(pred_str)

if __name__ == '__main__':
    main()