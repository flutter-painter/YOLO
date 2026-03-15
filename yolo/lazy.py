import sys
from pathlib import Path

import hydra
import torch
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    # Prefer CUDA when available (resolve "auto" to explicit GPU)
    accelerator = getattr(cfg, "accelerator", "auto")
    devices = getattr(cfg, "device", "auto")
    if (accelerator == "auto" or str(accelerator).lower() == "auto") and (
        devices == "auto" or str(devices).lower() == "auto"
    ):
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1  # use first GPU; override with device=[0,1] etc. for multi-GPU
        else:
            accelerator = "cpu"
            devices = 1

    # Use Tensor Cores on supported GPUs (e.g. RTX 4070) for better performance
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        # Avoid cuDNN "Plan failed / CUDNN_STATUS_NOT_SUPPORTED" warnings on some GPU/driver combos.
        torch.backends.cudnn.benchmark = False

    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        sync_batchnorm=True,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
