from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, QuantType, quantize_static as ort_quantize_static, CalibrationDataReader, CalibrationMethod, QuantFormat
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import read_dataset
from texo.data.processor import EvalMERImageProcessor
import json

MODEL_DIR = "./model/onnx"
DYNAMIC_QUANTIZE_DIR = "./model/onnx_quantized_dynamic"
STATIC_QUANTIZE_DIR = './model/onnx_quantized_static'

encode_model_path = f'{MODEL_DIR}/encoder_model.onnx'
decode_model_path = f'{MODEL_DIR}/decoder_model.onnx'
decoder_model_merged_path = f'{MODEL_DIR}/decoder_model_merged.onnx'
decoder_with_past_model_path = f'{MODEL_DIR}/decoder_with_past_model.onnx'

encode_quant_model_path = f'{DYNAMIC_QUANTIZE_DIR}/encoder_model.onnx'
decode_quant_model_path = f'{DYNAMIC_QUANTIZE_DIR}/decoder_model.onnx'
decoder_model_merged_quant_path = f'{DYNAMIC_QUANTIZE_DIR}/decoder_model_merged.onnx'
decoder_with_past_model_quant_path = f'{DYNAMIC_QUANTIZE_DIR}/decoder_with_past_model.onnx'

static_encode_quant_model_path = f'{STATIC_QUANTIZE_DIR}/encoder_model.onnx'
static_decode_quant_model_path = f'{STATIC_QUANTIZE_DIR}/decoder_model.onnx'
static_decoder_model_merged_quant_path = f'{STATIC_QUANTIZE_DIR}/decoder_model_merged.onnx'
static_decoder_with_past_model_quant_path = f'{STATIC_QUANTIZE_DIR}/decoder_with_past_model.onnx'

with open(os.path.join(MODEL_DIR, 'config.json'), 'r', encoding='utf-8') as _cf:
    _cfg = json.load(_cf)
DECODER_HEADS = int(_cfg.get('decoder', {}).get('decoder_attention_heads', 8))
DECODER_D_MODEL = int(_cfg.get('decoder', {}).get('d_model', 256))
DECODER_HEAD_DIM = max(1, DECODER_D_MODEL // max(1, DECODER_HEADS))

def quantize_dynamic():
    """
    对导出的 ONNX 模型执行动态量化（权重 INT8）。
    兼容不同版本的 ONNX Runtime：若不支持 `weight_type` 参数，则回退为默认配置。
    """
    import os

    os.makedirs(DYNAMIC_QUANTIZE_DIR, exist_ok=True)

    def _run(in_path: str, out_path: str):
        try:
            ort_quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)
        except TypeError:
            ort_quantize_dynamic(in_path, out_path)

    _run(decode_model_path, decode_quant_model_path)
    _run(encode_model_path, encode_quant_model_path)
    _run(decoder_model_merged_path, decoder_model_merged_quant_path)
    _run(decoder_with_past_model_path, decoder_with_past_model_quant_path)

    print('dynamic quantized models saved to', decode_quant_model_path, encode_quant_model_path)

def _quantize_static(model_path: str, output_path: str, encoder_model_path: str | None = None):
    """
    使用代表性图片进行静态量化校准，自动适配 Vision-Encoder-Decoder 的输入：
    - 对编码器：喂入 `pixel_values`。
    - 对解码器：同时喂入 `input_ids` 与通过编码器会话计算的 `encoder_hidden_states`。
    """

    class Vision2SeqCalibrationDataReader(CalibrationDataReader):
        def __init__(self, target_model_path: str, image_paths: list[str], enc_model_path: str | None, image_size={"height":384,"width":384}):
            self._sess = ort.InferenceSession(target_model_path, providers=["CPUExecutionProvider"])
            self._inputs = {i.name: i for i in self._sess.get_inputs()}
            self._processor = EvalMERImageProcessor(image_size=image_size)
            self._images = image_paths
            self._idx = 0
            self._enc_sess = ort.InferenceSession(enc_model_path, providers=["CPUExecutionProvider"]) if enc_model_path else None
            self._enc_input = self._enc_sess.get_inputs()[0].name if self._enc_sess else None

        def _zeros_like_shape(self, node_arg):
            shp = getattr(node_arg, 'shape', None)
            if not shp:
                return np.zeros((1, DECODER_HEADS, 0, DECODER_HEAD_DIM), dtype=np.float32)
            dims = []
            for d in shp:
                if isinstance(d, int):
                    dims.append(max(0, d))
                else:
                    s = str(d).lower()
                    if 'batch' in s:
                        dims.append(1)
                    elif 'head' in s:
                        dims.append(DECODER_HEADS)
                    elif 'length' in s or 'seq' in s:
                        dims.append(0)
                    elif 'embed' in s or 'dim' in s or 'hidden' in s:
                        dims.append(DECODER_HEAD_DIM)
                    else:
                        dims.append(1)
            if len(dims) == 0:
                dims = [1, DECODER_HEADS, 0, DECODER_HEAD_DIM]
            return np.zeros(tuple(dims), dtype=np.float32)

        def get_next(self):
            if self._idx >= len(self._images):
                return None
            img = Image.open(self._images[self._idx]).convert("RGB")
            self._idx += 1
            pixel = self._processor(img).unsqueeze(0).cpu().numpy().astype(np.float32)

            feed: dict[str, np.ndarray] = {}
            if "pixel_values" in self._inputs:
                feed["pixel_values"] = pixel
            need_enc = ("encoder_hidden_states" in self._inputs) or any(('.encoder.' in n) for n in self._inputs.keys())
            S = None
            enc_outs = None
            if need_enc and self._enc_sess and self._enc_input:
                enc_outs = self._enc_sess.run(None, {self._enc_input: pixel})
                enc_hidden = enc_outs[0].astype(np.float32)
                S = int(enc_hidden.shape[1])
                if "encoder_hidden_states" in self._inputs:
                    feed["encoder_hidden_states"] = enc_hidden
                if "encoder_attention_mask" in self._inputs and S is not None:
                    feed["encoder_attention_mask"] = np.ones((1, S), dtype=np.int64)
            if "input_ids" in self._inputs:
                feed["input_ids"] = np.array([[0]], dtype=np.int64)
            if "decoder_input_ids" in self._inputs:
                feed["decoder_input_ids"] = np.array([[0]], dtype=np.int64)
            if "attention_mask" in self._inputs:
                feed["attention_mask"] = np.array([[1]], dtype=np.int64)
            if "decoder_attention_mask" in self._inputs:
                feed["decoder_attention_mask"] = np.array([[1]], dtype=np.int64)
            if "position_ids" in self._inputs:
                feed["position_ids"] = np.array([[0]], dtype=np.int64)
            if "decoder_position_ids" in self._inputs:
                feed["decoder_position_ids"] = np.array([[0]], dtype=np.int64)
            if "use_cache_branch" in self._inputs:
                feed["use_cache_branch"] = np.array([False], dtype=np.bool_)
            for name, na in self._inputs.items():
                if ('past' in name) or ('past_key_values' in name) or name.endswith('.key') or name.endswith('.value'):
                    if S is not None and ('.encoder.' in name):
                        feed[name] = np.zeros((1, DECODER_HEADS, S, DECODER_HEAD_DIM), dtype=np.float32)
                    else:
                        feed[name] = self._zeros_like_shape(na)
            return feed

        def rewind(self):
            self._idx = 0

    pairs = read_dataset.read_dataset()
    image_paths = [p["img_path"] for p in pairs[:128]]

    data_reader = Vision2SeqCalibrationDataReader(model_path, image_paths, encoder_model_path)

    ort_quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True}
    )

    print("static quantized model saved to", output_path)

def quantize_static():
    """
    执行静态量化：编码器与解码器。
    解码器依赖编码器输出进行校准；跳过包含 past 的解码器变体以避免多输入复杂性。
    """
    os.makedirs(STATIC_QUANTIZE_DIR, exist_ok=True)

    _quantize_static(encode_model_path, static_encode_quant_model_path)
    _quantize_static(decode_model_path, static_decode_quant_model_path, encoder_model_path=encode_model_path)
    _quantize_static(decoder_model_merged_path, static_decoder_model_merged_quant_path, encoder_model_path=encode_model_path)
    _quantize_static(decoder_with_past_model_path, static_decoder_with_past_model_quant_path, encoder_model_path=encode_model_path)

if __name__ == '__main__':
    quantize_static()
    # quantize_dynamic()