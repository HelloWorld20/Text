"""
ONNX Runtime 稀疏加速脚本
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
try:
    from transformers import AutoConfig, AutoModel
    from texo.model.hgnet2 import HGNetv2Config, HGNetv2
    # 注册自定义视觉编码器到 transformers 的 Auto 系列
    AutoConfig.register("my_hgnetv2", HGNetv2Config)
    AutoModel.register(HGNetv2Config, HGNetv2)
except Exception:
    pass

# 可选注册，仅当安装了 optimum 时启用；未安装时不影响稀疏化流程
try:
    from optimum.exporters.tasks import TasksManager
    from optimum.exporters.onnx.model_configs import ViTOnnxConfig

    register_tasks_manager_onnx = TasksManager.create_register("onnx")

    @register_tasks_manager_onnx("my_hgnetv2", *["feature-extraction"])
    class HGNetv2OnnxConfig(ViTOnnxConfig):
        @property
        def inputs(self):
            return {"pixel_values": {0: "batch_size"}}
except Exception:
    register_tasks_manager_onnx = None


def _as_pretrained_dir(path: str) -> str:
    """将权重文件路径规整为 `from_pretrained` 可识别的目录路径。"""
    p = Path(path)
    if p.is_dir():
        return str(p)
    return str(p.parent)


def _export_onnx(pretrained_dir: str, output_dir: str, task: str = "image-to-text-with-past") -> None:
    """使用 Optimum 导出 VisionEncoderDecoder 到 ONNX（包含 encoder/decoder/trio）。

    若环境或模型类型不受支持，则优雅跳过导出，仅保留后续稀疏转换。
    """
    try:
        from optimum.exporters.onnx import main_export
    except Exception as e:
        print(f"Skip ONNX export: optimum unavailable: {e}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        main_export(
            pretrained_dir,
            task=task,
            output=out,
        )
    except Exception as e:
        print(f"Skip ONNX export due to error: {e}")
        return


def _compute_sparsity(arr: np.ndarray) -> float:
    """计算张量稀疏率（零元素比例）。"""
    total = arr.size
    if total == 0:
        return 0.0
    nnz = np.count_nonzero(arr)
    return float(1.0 - (nnz / total))


def _should_convert_initializer(name: str, arr: np.ndarray, min_sparsity: float) -> bool:
    """判断某个初始化权重是否应转换为稀疏张量。"""
    # print(arr.ndim)
    if arr.ndim < 2:
        return False
    # 放宽名称限制：不强制要求包含 "weight"，以兼容不同导出策略
    # print(name)
    
    sparsity = _compute_sparsity(arr)
    # print('_should_convert_initializer:::::::::::::')
    # print(sparsity >= min_sparsity, sparsity, min_sparsity)
    return sparsity >= min_sparsity


def _to_sparse_tensor_proto(init_name: str, dense: np.ndarray):
    """将 Dense numpy 数组转换为 SparseTensorProto。

    兼容老版本 onnx（SparseTensorProto 无 name 字段）。若无法设置 name，返回 None。
    """
    import onnx
    from onnx import helper, numpy_helper

    nz = np.nonzero(dense)
    if len(nz[0]) == 0:
        values = np.asarray([], dtype=dense.dtype)
        indices = np.empty((0, dense.ndim), dtype=np.int64)
    else:
        indices = np.vstack(nz).T.astype(np.int64)
        values = dense[nz].astype(dense.dtype)

    values_tp = numpy_helper.from_array(values, name=f"{init_name}__values")
    indices_tp = numpy_helper.from_array(indices, name=f"{init_name}__indices")
    sparse = helper.make_sparse_tensor(values_tp, indices_tp, list(dense.shape))
    try:
        # 新版 onnx 支持为 SparseTensorProto 设置 name，用于 graph.sparse_initializer 引用
        sparse.name = init_name
    except AttributeError:
        # 旧版 onnx 无 name 字段，返回 None 表示不做转换
        return None
    return sparse


def convert_onnx_initializers_to_sparse(
    model_path: str,
    min_sparsity: float = 0.5,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """将 ONNX 模型中的稀疏权重转换为 `sparse_initializer`。

    返回值为 (候选数量, 实际转换数量)。当 `dry_run=True` 时不写回文件。
    """
    import onnx
    from onnx import numpy_helper
    model = onnx.load(model_path)
    graph = model.graph

    candidates: List[Tuple[str, np.ndarray]] = []
    for init in list(graph.initializer):
        arr = numpy_helper.to_array(init)
        if _should_convert_initializer(init.name, arr, min_sparsity):
            candidates.append((init.name, arr))

    # 同时扫描 Constant 节点（旧导出可能将权重存为 Constant.value）
    const_candidates: List[Tuple[int, str, np.ndarray]] = []  # (node_index, output_name, array)
    for idx, node in enumerate(graph.node):
        if node.op_type != "Constant":
            continue
        # 找到 dense Tensor 值
        value_attr = None
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                value_attr = attr
                break
        if value_attr is None:
            continue
        arr = numpy_helper.to_array(value_attr.t)
        out_name = node.output[0] if len(node.output) > 0 else f"const_{idx}"
        if _should_convert_initializer(out_name, arr, min_sparsity):
            const_candidates.append((idx, out_name, arr))

    converted = 0
    if not dry_run:
        # 先构建可转换的稀疏权重列表，避免删除后无法恢复
        to_remove = set()
        sparse_list = []
        constant_nodes = []
        for name, arr in candidates:
            # 优先尝试使用 sparse_initializer（需要 name 字段）
            sparse = _to_sparse_tensor_proto(name, arr)
            if sparse is None:
                # 回退：使用 Constant 节点的 sparse_value 属性，不需要 name 字段
                import onnx
                from onnx import helper, numpy_helper
                nz = np.nonzero(arr)
                if len(nz[0]) == 0:
                    values = np.asarray([], dtype=arr.dtype)
                    indices = np.empty((0, arr.ndim), dtype=np.int64)
                else:
                    indices = np.vstack(nz).T.astype(np.int64)
                    values = arr[nz].astype(arr.dtype)

                values_tp = numpy_helper.from_array(values, name=f"{name}__values")
                indices_tp = numpy_helper.from_array(indices, name=f"{name}__indices")
                sparse_tensor = helper.make_sparse_tensor(values_tp, indices_tp, list(arr.shape))
                const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[name],
                    sparse_value=sparse_tensor,
                )
                constant_nodes.append(const_node)
                to_remove.add(name)
            else:
                sparse_list.append(sparse)
                to_remove.add(name)

        # 删除对应的 dense initializer
        for i in range(len(graph.initializer) - 1, -1, -1):
            if graph.initializer[i].name in to_remove:
                del graph.initializer[i]

        # 添加稀疏 initializer
        for sparse in sparse_list:
            st = graph.sparse_initializer.add()
            st.CopyFrom(sparse)
            converted += 1

        # 添加 Constant 节点以承载稀疏张量（旧版 onnx）
        for node in constant_nodes:
            graph.node.append(node)
            converted += 1

        # 将已有 Constant 节点的 dense value 替换为 sparse_value
        if len(const_candidates) > 0:
            from onnx import helper
            for idx, out_name, arr in const_candidates:
                node = graph.node[idx]
                nz = np.nonzero(arr)
                if len(nz[0]) == 0:
                    values = np.asarray([], dtype=arr.dtype)
                    indices = np.empty((0, arr.ndim), dtype=np.int64)
                else:
                    indices = np.vstack(nz).T.astype(np.int64)
                    values = arr[nz].astype(arr.dtype)
                values_tp = numpy_helper.from_array(values, name=f"{out_name}__values")
                indices_tp = numpy_helper.from_array(indices, name=f"{out_name}__indices")
                sparse_tensor = helper.make_sparse_tensor(values_tp, indices_tp, list(arr.shape))
                del node.attribute[:]
                node.attribute.extend([helper.make_attribute("sparse_value", sparse_tensor)])
                converted += 1

        onnx.save(model, model_path)
    else:
        # dry_run 模式下返回候选数量作为 converted 统计
        converted = len(candidates) + len(const_candidates)

    return len(candidates) + len(const_candidates), converted


def convert_directory_models(
    onnx_dir: str,
    min_sparsity: float = 0.5,
    dry_run: bool = False,
) -> List[Tuple[str, Tuple[int, int]]]:
    """批量转换目录中的 trio ONNX 模型，返回每个文件的统计信息。"""
    targets = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
        "decoder_model_merged.onnx",
    ]
    results: List[Tuple[str, Tuple[int, int]]] = []
    for name in targets:
        p = Path(onnx_dir).joinpath(name)
        if p.exists():
            stat = convert_onnx_initializers_to_sparse(str(p), min_sparsity=min_sparsity, dry_run=dry_run)
            results.append((name, stat))
    return results


def main():
    """命令行入口：从剪枝权重导出 ONNX 并应用 SparseOps。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        default="/Users/leon.w/workspace/cityu/6009/Texo/model/pruning/model.safetensors",
        help="剪枝后的权重文件或所在目录",
    )
    parser.add_argument(
        "--output",
        default="/Users/leon.w/workspace/cityu/6009/Texo/model/onnx_sparse",
        help="ONNX 导出与稀疏化输出目录",
    )
    parser.add_argument(
        "--task",
        default="image-to-text-with-past",
        help="Optimum 导出任务标识",
    )
    parser.add_argument(
        "--min-sparsity",
        type=float,
        default=0.5,
        help="仅转换稀疏率不低于该阈值的权重",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="跳过 ONNX 导出，仅对现有 ONNX 做稀疏化",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计可转换数量，不写回模型文件",
    )

    args = parser.parse_args()

    pretrained_dir = _as_pretrained_dir(args.pretrained)
    out_dir = args.output

    if not args.skip_export:
        _export_onnx(pretrained_dir, out_dir, task=args.task)

    stats = convert_directory_models(out_dir, min_sparsity=args.min_sparsity, dry_run=args.dry_run)
    for fname, (cand, conv) in stats:
        print(f"{fname}: candidates={cand}, converted={conv}")

    """
    原始体积：
    decoder_model_merged.onnx: 25.9 MB
    decoder_model.onnx: 25.9 MB
    decoder_with_past_model.onnx: 20.3 MB
    encoder_model.onnx: 54.2 MB

    90% 稀疏率
    模型生成体积为：
    decoder_model_merged.onnx: 13.1 MB
    decoder_model.onnx: 13 MB
    decoder_with_past_model.onnx: 10.2 MB
    encoder_model.onnx: 48.8 MB

    """

if __name__ == "__main__":
    main()

