import copy
import logging
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from datamodule import MERDataModule
from torch.utils.data import DataLoader
from texo.data.dataset import MERDataset
from texo.data.processor import TrainMERImageProcessor, EvalMERImageProcessor, TextProcessor
from task import FormulaNetLit
from texo.utils.config import DictConfig, OmegaConf, hydra


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """按幅度进行细粒度剪枝并返回二值掩码"""
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
    mask = torch.gt(importance, threshold)
    tensor.mul_(mask)
    return mask


class FineGrainedPruner:
    def __init__(self, model: torch.nn.Module, sparsity_dict: float | Dict[str, float]):
        """初始化剪枝器并为模型生成掩码"""
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        """将掩码应用到模型参数以保持稀疏"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model: torch.nn.Module, sparsity_dict: float | Dict[str, float]) -> Dict[str, torch.Tensor]:
        """为模型中可剪枝参数生成二值掩码"""
        masks: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.dim() > 1:
                if isinstance(sparsity_dict, dict):
                    masks[name] = fine_grained_prune(param, sparsity_dict.get(name, 0.0))
                else:
                    assert 0.0 <= sparsity_dict < 1.0
                    if sparsity_dict > 0:
                        masks[name] = fine_grained_prune(param, sparsity_dict)
        return masks


class MaskingCallback(Callback):
    def __init__(self, masks: Dict[str, torch.Tensor]):
        """Lightning 回调：在优化器步前后应用掩码并屏蔽梯度"""
        super().__init__()
        self.masks = masks

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        model = pl_module.model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks and param.grad is not None:
                    param.grad.mul_(self.masks[name])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        model = pl_module.model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks:
                    param.mul_(self.masks[name])


def _detect_accelerator_and_devices() -> tuple[str, int]:
    """根据本机环境选择加速器与设备数量"""
    if torch.cuda.is_available():
        return "gpu", 1
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


@hydra.main(version_base="1.3.2", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    """运行 LTH 初级版：预训练→剪枝→回滚→稀疏微调"""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    # 解决插值：将 tokenizer_path 显式写入 data_config
    try:
        if isinstance(data_config, dict) and "text_processor" in data_config:
            data_config["text_processor"]["tokenizer_path"] = model_config.get("tokenizer_path", data_config["text_processor"].get("tokenizer_path"))
    except Exception:
        pass

    # 若预训练权重不存在，则从随机初始化开始
    try:
        import os
        ckpt_path = model_config.get("pretrained", "")
        if isinstance(ckpt_path, str) and ckpt_path and not os.path.exists(ckpt_path):
            model_config["pretrained"] = ""
    except Exception:
        model_config["pretrained"] = ""

    lit = FormulaNetLit(model_config, training_config)
    # 主数据模块（HF datasets）；如不可用则回退到 simple 本地数据集
    def _build_simple_dm(dc: Dict) -> L.LightningDataModule:
        class SimpleDM(L.LightningDataModule):
            def __init__(self, data_config: Dict):
                super().__init__()
                self.dc = data_config
            def setup(self, stage=None):
                tp = TextProcessor(self.dc["text_processor"])  # 复用同一 tokenizer
                train_ip = TrainMERImageProcessor(**self.dc["image_processor"])  #type: ignore
                eval_ip = EvalMERImageProcessor(**self.dc["image_processor"])    #type: ignore
                img_dir = "./data/dataset/simple/images"
                txt_path = "./data/dataset/simple/train.txt"
                self.train_dataset = MERDataset(img_dir, txt_path, train_ip, tp)
                self.val_dataset = MERDataset(img_dir, txt_path, eval_ip, tp)
            def train_dataloader(self):
                return DataLoader(self.train_dataset,
                                  batch_size=min(self.dc.get("train_batch_size", 8), 8),
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=self.train_dataset.collate_fn)
            def val_dataloader(self):
                return DataLoader(self.val_dataset,
                                  batch_size=min(self.dc.get("val_batch_size", 8), 8),
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=self.val_dataset.collate_fn)
        return SimpleDM(dc)

    try:
        import os
        hf_train_root = data_config.get("train_dataset_path", "")
        use_hf = isinstance(hf_train_root, str) and os.path.exists(hf_train_root)
    except Exception:
        use_hf = False

    datamodule = MERDataModule(data_config) if use_hf else _build_simple_dm(data_config)

    init_state = copy.deepcopy(lit.model.state_dict())

    acc, devs = _detect_accelerator_and_devices()

    pretrain_epochs = int(getattr(cfg, "lth", {}).get("pretrain_epochs", 1))
    sparsity = float(getattr(cfg, "lth", {}).get("sparsity", 0.8))
    finetune_epochs = int(getattr(cfg, "lth", {}).get("finetune_epochs", 1))

    logging.log(logging.INFO, f"[LTH] Dense pretrain epochs={pretrain_epochs}, sparsity={sparsity}, finetune epochs={finetune_epochs}")

    trainer_dense = hydra.utils.instantiate(
        cfg.trainer,
        profiler=None,
        max_epochs=pretrain_epochs,
        accelerator=acc,
        devices=devs,
        val_check_interval=1 if not use_hf else cfg.trainer.get("val_check_interval", 1000),
    )
    trainer_dense.fit(lit, datamodule=datamodule)

    pruner = FineGrainedPruner(lit.model, sparsity)
    masks = pruner.masks

    with torch.no_grad():
        for name, param in lit.model.named_parameters():
            if name in init_state:
                param.copy_(init_state[name])
            if name in masks:
                param.mul_(masks[name])

    existing_callbacks = []
    if hasattr(cfg.trainer, "callbacks"):
        for cb_conf in cfg.trainer.callbacks:
            existing_callbacks.append(hydra.utils.instantiate(cb_conf))
    existing_callbacks.append(MaskingCallback(masks))

    trainer_sparse = hydra.utils.instantiate(
        cfg.trainer,
        profiler=None,
        max_epochs=finetune_epochs,
        accelerator=acc,
        devices=devs,
        callbacks=existing_callbacks,
        val_check_interval=1 if not use_hf else cfg.trainer.get("val_check_interval", 1000),
    )
    trainer_sparse.fit(lit, datamodule=datamodule)


if __name__ == "__main__":
    main()