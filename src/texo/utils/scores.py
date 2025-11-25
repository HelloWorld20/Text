import evaluate
from rapidfuzz.distance import Levenshtein

bleu = None

def compute_bleu(pred_str: list[str], ref_str: list[str]):
    """计算 BLEU 分数，首次调用时延迟加载 evaluate 模块"""
    global bleu
    if bleu is None:
        try:
            bleu = evaluate.load("bleu", keep_in_memory=False)
        except Exception:
            return 0.0
    results = bleu.compute(
        predictions=pred_str, references=ref_str, tokenizer=lambda x: x.split(" ")
    )
    return results["bleu"]

def compute_edit_distance(pred_str: list[str], ref_str: list[str]):
    """计算归一化编辑距离"""
    results = [Levenshtein.normalized_distance(p, r, processor=str.split) for p, r in zip(pred_str, ref_str)]
    return sum(results)/len(results)

if __name__ == '__main__':
    pred_str = "hello world"
    ref_str = "hello world"
    pred_str2 = "[ C ] h t g k b S b"
    ref_str2 = "[ C ] h t a b S b"
    assert compute_bleu([pred_str], [ref_str]) == 0
    assert compute_bleu([pred_str2], [ref_str2]) == 0.5253819788848316
    assert compute_edit_distance([pred_str], [ref_str]) == 0
    assert compute_edit_distance([pred_str2], [ref_str2]) == 0.2