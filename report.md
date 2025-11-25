    原始文件体积：
    decoder_model_merged.onnx: 25.9 MB
    decoder_model.onnx: 25.9 MB
    decoder_with_past_model.onnx: 20.3 MB
    encoder_model.onnx: 54.2 MB


# 非结构化剪枝+Onnx Sparse Ops

    python run_benchmark.py --compressed_dir ./model/onnx_sparse

    **inference speed is very slow**


    90% 稀疏率
    模型生成体积为：
    decoder_model_merged.onnx: 13.1 MB
    decoder_model.onnx: 13 MB
    decoder_with_past_model.onnx: 10.2 MB
    encoder_model.onnx: 48.8 MB

    100条数据

    result：
    Average MSE(BEFORE vs ORIGINAL): 0.7154
    Average MSE(AFTER vs ORIGINAL): 0.9251
    Average MSE(BEFORE vs AFTER): 0.9233


# quantization

    python run_quantize.py

## dynamic quantize

    模型体积
    decoder_model_merged.onnx: 26 MB
    decoder_model.onnx: 6.7 MB
    decoder_with_past_model.onnx: 5.2 MB
    encoder_model.onnx: 13.8 MB

100条数据：
Average MSE(FP32 vs GT): 0.7154
Average MSE(INT8 vs GT): 0.7322
Average MSE(FP32 vs INT8): 0.4114

## static quantize

    模型体积
    decoder_model_merged.onnx: 26 MB
    decoder_model.onnx: 6.7 MB
    decoder_with_past_model.onnx: 5.3 MB
    encoder_model.onnx: 13.8 MB

100条数据：
Average MSE(FP32 vs GT): 0.7154
Average MSE(INT8 vs GT): 0.7257
Average MSE(FP32 vs INT8): 0.4230