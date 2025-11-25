import read_dataset
import benchmark
from tqdm import tqdm
import argparse
import time


MODEL_DIR = "./model/onnx"
# QUANTIZED_DIR = "./model/onnx_quantized_dynamic"
COMPRESED_DIR = "./model/onnx_quantized_static"

def compare_levenshtein_distance():
    pairs = read_dataset.read_dataset()

    sample = pairs[:1]

    total_cer = 0.0
    total_t_fp32 = 0.0
    total_t_int8 = 0.0

    for pair in tqdm(sample, desc="Processing samples"):
        # img_path = pair["img_path"]
        # text = pair["text"]
        # print(f"img_path: {img_path}")
        # print(f"text: {text}")
        # print("================================")

        cer, t_fp32, t_int8 = benchmark.compare_levenshtein_distance(
            fp32_dir=MODEL_DIR,
            int8_dir=COMPRESED_DIR,
            image_path=pair["img_path"]
        )

        total_cer += cer
        total_t_fp32 += t_fp32
        total_t_int8 += t_int8

        # is_accuracy_same = benchmark.compare_accuracy(
        #     fp32_dir=MODEL_DIR,
        #     int8_dir=QUANTIZED_DIR,
        #     image_path=pair["img_path"]
        # )


        # print(f"Is accuracy same: {is_accuracy_same}")
        
    print(f"Average CER: {total_cer / len(sample):.4f}")
    print(f"Average FP32 latency: {total_t_fp32 / len(sample):.4f}s")
    print(f"Average INT8 latency: {total_t_int8 / len(sample):.4f}s")
    print("================================")
    

def compare_accuracy():
    pairs = read_dataset.read_dataset()
    sample = pairs[:10]
    misses = 0

    for pair in tqdm(sample, desc="Processing samples"):

        equal = benchmark.compare_accuracy(
            fp32_dir=MODEL_DIR,
            int8_dir=COMPRESED_DIR,
            image_path=pair["img_path"]
        )

        if (not equal):
            misses += 1       

    print(f"Misses: {misses}")
    print(f"Accuracy: {1 - misses / len(sample):.4f}")
    print("================================")

def default_accuracy():
    """
    未量化的模型与真实文本的准确率
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    misses = 0

    for pair in tqdm(sample, desc="Processing samples"):

        equal = benchmark.original_accuracy(
            fp32_dir=MODEL_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        if (not equal):
            misses += 1       

    print(f"Misses: {misses}")
    print(f"Accuracy: {1 - misses / len(sample):.4f}")
    print("================================")
    # 跑100个数据，准确率为0

def default_distance():
    """
    未量化的模型与真实文本的距离
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    total_distance = 0.0

    for pair in tqdm(sample, desc="Processing samples"):

        distance = benchmark.original_distance(
            fp32_dir=MODEL_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        total_distance += distance       

    print(f"Average Distance: {total_distance / len(sample):.4f}")
    print("================================")
    # 跑100个数据，距离为Average Distance: 0.1258

def quantized_distance():
    """
    量化后的模型与真实文本的距离
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    total_distance = 0.0

    for pair in tqdm(sample, desc="Processing samples"):

        distance = benchmark.original_distance(
            fp32_dir=COMPRESED_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        total_distance += distance       

    print(f"Average Distance: {total_distance / len(sample):.4f}")
    print("================================")
    # 跑100个数据，距离为Average Distance: 0.1407

def compare_mse():
    args = parseArgs()
    """
    对比均方误差（MSE）：
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:args.sample_size]
    total_mse_fp32_gt = 0.0
    total_mse_int8_gt = 0.0
    total_mse_fp32_int8 = 0.0

    start_time = time.time()
    for pair in tqdm(sample, desc="Processing samples"):
        mse_fp32_gt, mse_int8_gt, mse_fp32_int8, _, _ = benchmark.compare_mse(
            fp32_dir=args.model_dir,
            int8_dir=args.compressed_dir,
            image_path=pair["img_path"],
            text=pair["text"]
        )
        total_mse_fp32_gt += mse_fp32_gt
        total_mse_int8_gt += mse_int8_gt
        total_mse_fp32_int8 += mse_fp32_int8

    
        # print(f"MSE(FP32 vs GT): {mse_fp32_gt:.4f}")
        # print(f"MSE(INT8 vs GT): {mse_int8_gt:.4f}")
        # print(f"MSE(FP32 vs INT8): {mse_fp32_int8:.4f}")
        # print("================================")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f}s")
    
    print(f"Average MSE(BEFORE vs ORIGINAL): {total_mse_fp32_gt / len(sample):.4f}")
    print(f"Average MSE(AFTER vs ORIGINAL): {total_mse_int8_gt / len(sample):.4f}")
    print(f"Average MSE(BEFORE vs AFTER): {total_mse_fp32_int8 / len(sample):.4f}")
    print("================================")
   
def parseArgs():
    parser = argparse.ArgumentParser(description="Quantize model")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to the model directory")
    parser.add_argument("--compressed_dir", type=str, default=COMPRESED_DIR, help="Path to the compressed directory")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to use for evaluation")
    args = parser.parse_args()
    print('arguments: ')
    print(args)
    return args

if __name__ == "__main__":
    
    # compare_levenshtein_distance()
    # compare_accuracy()
    # default_accuracy()
    # default_distance()
    # quantized_distance()
    compare_mse()

# dynamic quantize
# 100条数据：
# Average MSE(FP32 vs GT): 0.7154
# Average MSE(INT8 vs GT): 0.7322
# Average MSE(FP32 vs INT8): 0.4114

# static quantize
# 100条数据：
# Average MSE(FP32 vs GT): 0.7154
# Average MSE(INT8 vs GT): 0.7257
# Average MSE(FP32 vs INT8): 0.4230
    

    