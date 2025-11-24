import time
import Levenshtein
from onnxruntime.quantization import quantize_dynamic
from optimum.onnxruntime import ORTModelForVision2Seq
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from PIL import Image
import onnxruntime as ort

ort.set_default_logger_severity(3)

device = "cuda" if ort.get_device() == "GPU" else "cpu"

def generate_text(model_dir, image_path):
    model = ORTModelForVision2Seq.from_pretrained(model_dir)
    model.to(device)
    processor = EvalMERImageProcessor(image_size={"height": 384, "width": 384})
    text_processor = TextProcessor(config={
        "tokenizer_path": "data/tokenizer",
        "tokenizer_config":{
            "add_special_tokens": True,
            "max_length": 1024,
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
            "return_attention_mask": False,
        }
    })

    pixel_values = processor(Image.open(image_path).convert("RGB")).unsqueeze(0)

    t0 = time.time()
    outputs = model.generate(pixel_values)
    latency = time.time() - t0

    text = text_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, latency


def compare_levenshtein_distance(fp32_dir, int8_dir, image_path):
    """
    Levenshtein Distance 算法是一种计算两个字符串之间的编辑距离的算法
    """
    text_fp32, t_fp32 = generate_text(fp32_dir, image_path)
    text_int8, t_int8 = generate_text(int8_dir, image_path)
    print('--------------------')
    print(text_fp32)
    print('-----------------')
    print(text_int8)

    # text_int8 = text_int8 + ' '

    cer = Levenshtein.distance(text_fp32, text_int8) / len(text_fp32)

    # print("\n========== 比较结果 ==========")
    # print("FP32 输出：", text_fp32)
    # print("INT8 输出：", text_int8)
    # print(f"CER: {cer:.4f}")
    # print(f"FP32 latency: {t_fp32:.4f}s")
    # print(f"INT8 latency: {t_int8:.4f}s")
    # print("================================")

    return cer, t_fp32, t_int8

def compare_accuracy(fp32_dir, int8_dir, image_path):
    """
    比较 FP32 和 INT8 模型的输出是否完全一致
    """
    text_fp32, _ = generate_text(fp32_dir, image_path)
    text_int8, _ = generate_text(int8_dir, image_path)

    return text_fp32 == text_int8

def original_accuracy(fp32_dir, image_path, text):
    """
    计算 FP32 模型的原始准确率
    """
    text_fp32, _ = generate_text(fp32_dir, image_path)
    print(f"Predicted Text::::::: {text_fp32}")
    print(f"Actual Text::::::: {text}")
    return text_fp32 == text

def original_distance(fp32_dir, image_path, text):
    """
    计算 指定 模型的与真实文本的距离
    """
    text_fp32, _ = generate_text(fp32_dir, image_path)
    return Levenshtein.distance(text_fp32, text) / len(text_fp32)


def mean_square_error_text(a: str, b: str) -> float:
    """
    计算两个文本之间的“均方误差（MSE）”。

    定义与实现说明：
    - 按字符级对齐到相同长度；长度不一致时，用空字符（"\0"）右侧填充。
    - 每个位置的误差 e 为 0/1：字符相同记 0，不同记 1；由于 e∈{0,1}，平方 e^2 与 e 相同。
    - MSE = 所有位置误差的平方的平均值，等价于归一化的 Hamming 距离。

    注意：这是对文本的 MSE 近似定义，适合快速对比字符级一致性；
    若需更严格的文本相似度，建议结合 CER/Levenshtein 距离一起参考。
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    a_padded = a.ljust(max_len, "\0")
    b_padded = b.ljust(max_len, "\0")
    errors = (0 if a_padded[i] == b_padded[i] else 1 for i in range(max_len))
    total = 0
    count = 0
    for e in errors:
        total += (e * e)
        count += 1
    return total / count if count else 0.0


def compare_mse(fp32_dir: str, int8_dir: str, image_path: str, text: str):
    """
    对比均方误差（MSE）：
    - 输出量化前（FP32）模型与真实文本（GT）的 MSE
    - 输出量化后（INT8）模型与真实文本（GT）的 MSE
    - 输出量化前后（FP32 vs INT8）模型输出之间的 MSE

    同时返回两种模型的推理延迟（latency）。
    """
    text_fp32, t_fp32 = generate_text(fp32_dir, image_path)
    text_int8, t_int8 = generate_text(int8_dir, image_path)

    mse_fp32_gt = mean_square_error_text(text_fp32, text)
    mse_int8_gt = mean_square_error_text(text_int8, text)
    mse_fp32_int8 = mean_square_error_text(text_fp32, text_int8)

    # print("\n========== MSE 对比 ==========")
    # print("GT 文本：", text)
    # print("FP32 输出：", text_fp32)
    # print("INT8 输出：", text_int8)
    # print(f"MSE(FP32 vs GT): {mse_fp32_gt:.4f}")
    # print(f"MSE(INT8 vs GT): {mse_int8_gt:.4f}")
    # print(f"MSE(FP32 vs INT8): {mse_fp32_int8:.4f}")
    # print(f"FP32 latency: {t_fp32:.4f}s")
    # print(f"INT8 latency: {t_int8:.4f}s")
    # print("================================")

    return mse_fp32_gt, mse_int8_gt, mse_fp32_int8, t_fp32, t_int8


MODEL_DIR = "./model/onnx"
QUANTIZED_DIR = "./model/onnx_quantized"

if __name__ == "__main__":
    compare_levenshtein_distance(
        fp32_dir=MODEL_DIR,
        int8_dir=QUANTIZED_DIR,
        image_path="./TechnoSelection/test_img/多行公式.png"
    )