import logging
import torch.distributed as dist
from typing import Optional
from pathlib import Path
from datetime import datetime

def is_main_process() -> bool:
    """检查是否是主进程(rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_logging_handler(
    log_dir: Path,
    log_filename_prefix: str = "training",
    level: int = logging.INFO,
    level_non_main: int = logging.WARNING,
) -> None:
    """
    Configures basic logging for a distributed training script.

    Sets up file and stream handlers for the root logger on the main process,
    and configures a simpler logging setup for other processes.

    Args:
        log_dir (Path): The directory to save log files in.
        log_filename_prefix (str): Prefix for the log file name.
        level (int): Logging level for the main process.
        level_non_main (int): Logging level for non-main processes.
    """
    if is_main_process():
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{log_filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Clean up root logger handlers before applying basicConfig
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        # Clean up for non-main processes as well to ensure clean setup
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logging.basicConfig(
            level=level_non_main,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

def get_dist_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取分布式训练的logger
    
    Args:
        name: logger名称
        
    Returns:
        logging.Logger: 配置好的logger实例
    """
    logger = logging.getLogger(name)
    
    # 只有在分布式环境且不是主进程时才禁用日志输出
    if dist.is_initialized() and not is_main_process():
        logger.disabled = True
    
    return logger

import wandb
from typing import Dict, Any, Tuple

def setup_wandb_recovery(config, is_main_process: bool = True) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Setup W&B run recovery logic based on configuration.
    
    Args:
        config: ViralBERTConfig instance
        is_main_process: Whether this is the main process
        
    Returns:
        Tuple of (wandb_run_id, wandb_resume_policy, is_resuming_wandb)
    """
    wandb_run_id = None
    wandb_resume_policy = None
    is_resuming_wandb = False

    # Only attempt to resume W&B run in 'recovery' mode
    if config.wandb_enabled and config.resume_from_checkpoint and config.resume_mode == 'recovery':
        wandb_id_path = Path(config.resume_from_checkpoint) / ".wandb_run_id"
        if wandb_id_path.exists():
            wandb_run_id = wandb_id_path.read_text().strip()
            wandb_resume_policy = "allow"  # 使用 'allow' 以便在无法恢复时创建新 run
            is_resuming_wandb = True
            if is_main_process:
                logger = logging.getLogger(__name__)
                logger.info(f"W&B Recovery: Attempting to resume run ID '{wandb_run_id}'.")
        else:
            if is_main_process:
                logger = logging.getLogger(__name__)
                logger.warning("W&B Recovery: '.wandb_run_id' file not found in checkpoint. A new run will be created.")
    # 在 'epoch_extend' 和 'adaptation_extend' 模式下，我们总是创建一个新的 run
    
    return wandb_run_id, wandb_resume_policy, is_resuming_wandb


def initialize_wandb(config, wandb_run_id: Optional[str], wandb_resume_policy: Optional[str], 
                    is_resuming_wandb: bool, is_main_process: bool = True) -> bool:
    """
    Initialize W&B and sync configuration.
    
    Args:
        config: ViralBERTConfig instance 
        wandb_run_id: W&B run ID for recovery
        wandb_resume_policy: Resume policy ("allow", "must", etc.)
        is_resuming_wandb: Whether attempting to resume
        is_main_process: Whether this is the main process
        
    Returns:
        bool: Updated is_resuming_wandb status
    """
    if not is_main_process or not config.wandb_enabled:
        return is_resuming_wandb
        
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        group=config.wandb_group,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=config.to_dict(),  # 将基础配置传递给 wandb
        id=wandb_run_id,
        resume=wandb_resume_policy,
    )
    
    logger = logging.getLogger(__name__)
    
    # 检查恢复是否成功
    if is_resuming_wandb and wandb.run and not wandb.run.resumed:
        logger.warning(
            f"W&B Recovery: Failed to resume run ID '{wandb_run_id}'. "
            f"The original run may be in 'killed' state or unavailable. "
            f"A new W&B run '{wandb.run.id}' has been created instead."
        )
        # 因为我们创建了一个新 run，所以不再是"恢复"状态
        is_resuming_wandb = False
    elif is_resuming_wandb and wandb.run and wandb.run.resumed:
        logger.info(f"W&B Recovery: Successfully resumed run ID '{wandb_run_id}'.")
    
    # 用 wandb.config 中的 sweep 参数覆盖我们的配置对象
    # 移除 hasattr 检查，以允许 sweep 注入新的参数 (例如 max_steps_for_sweep)
    for key, value in wandb.config.items():
        if not key.startswith("_"):  # 过滤掉 wandb 的内部键
            setattr(config, key, value)
            
    return is_resuming_wandb


def generate_run_name(config, is_main_process: bool = True) -> str:
    """
    Generate unique run name for training.
    
    Args:
        config: ViralBERTConfig instance
        is_main_process: Whether this is the main process
        
    Returns:
        str: Generated run name
    """
    run_name = config.run_name
    
    if is_main_process:
        if config.wandb_enabled and wandb.run and wandb.run.sweep_id:
            # For sweep runs, create a unique run_name to avoid output directory conflicts
            # by appending the unique run ID to the base name from the config.
            base_name = config.run_name or "sweep"
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_name}-{wandb.run.id}"
            config.run_name = run_name # Update config for logging purposes
        else:
            # For regular runs, use the specified name or generate a timestamped one.
            run_name = config.run_name or f"viralbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        # For runs with wandb disabled
        run_name = config.run_name or f"viralbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return run_name


def save_wandb_run_id(output_dir: Path, is_main_process: bool = True):
    """
    Save W&B run ID to checkpoint directory for recovery.
    
    Args:
        output_dir: Output directory path
        is_main_process: Whether this is the main process
    """
    if is_main_process and wandb.run:
        wandb_id_file = output_dir / ".wandb_run_id"
        wandb_id_file.write_text(wandb.run.id)
        logger = logging.getLogger(__name__)
        logger.info(f"Saved new W&B run ID '{wandb.run.id}' to checkpoint directory.")


def sync_wandb_step(global_step: int, is_resuming_wandb: bool, scheduler, is_main_process: bool = True) -> int:
    """
    Synchronize W&B step with checkpoint step in recovery mode.
    
    Args:
        global_step: Current global step from checkpoint
        is_resuming_wandb: Whether resuming W&B
        scheduler: Learning rate scheduler
        is_main_process: Whether this is the main process
        
    Returns:
        int: Updated global step
    """
    if not (is_main_process and is_resuming_wandb and wandb.run and wandb.run.resumed):
        return global_step
        
    logger = logging.getLogger(__name__)
    wandb_step = wandb.run.step
    if wandb_step > global_step:
        steps_to_advance = wandb_step - global_step
        logger.warning(
            f"W&B step ({wandb_step}) is ahead of checkpoint step ({global_step}). "
            f"Syncing to W&B step to prevent logging conflicts."
        )
        global_step = wandb_step
        
        # Also advance the scheduler to match
        if scheduler:
            logger.info(f"Advancing scheduler by {steps_to_advance} steps to match W&B...")
            for _ in range(steps_to_advance):
                scheduler.step()
                
    elif wandb_step < global_step:
        logger.warning(
            f"Checkpoint step ({global_step}) is ahead of W&B step ({wandb_step}). "
            f"This can happen if training continued after the last W&B sync. Using checkpoint step."
        )
        
    return global_step


class WandbLogger:
    """Weights & Biases 日志记录器"""
    def __init__(
        self,
        config,
        is_main_process: bool = True,
        project: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        save_code: bool = False,
        log_artifacts: bool = False,
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
        use_existing_run: bool = False,
    ):
        self.enabled = config.wandb_enabled and is_main_process
        if not self.enabled:
            return
            
        # 只有当不使用现有 run 或现有 run 不存在时才初始化
        if not use_existing_run or wandb.run is None:
            wandb.init(
                project=project or config.wandb_project,
                name=name or config.wandb_name,
                group=group or config.wandb_group,
                tags=tags or config.wandb_tags,
                notes=notes or config.wandb_notes,
                config=config.to_dict(),
                save_code=save_code,
                resume=resume,
                id=run_id,
            )
        # Use the existing logger from this module
        logger = logging.getLogger(__name__)
        if wandb.run:
            logger.info(f"WandbLogger is attached to run: {wandb.run.name}")
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """记录指标"""
        if not self.enabled:
            return
        wandb.log(metrics, step=step, commit=commit)
        
    def log_model_graph(self, model, log_freq=1000):
        """记录模型结构
        
        Args:
            model: 模型实例
            log_freq: 记录梯度的频率，设置为None禁用梯度记录
        """
        if not self.enabled:
            return
        logger = logging.getLogger(__name__)
        try:
            wandb.watch(
                model,
                log="gradients",  # 或者使用 "all" 记录参数和梯度
                log_freq=log_freq,  # 每隔多少步记录一次
                idx=0,  # 用于多个模型时的索引
                log_graph=True  # 是否记录计算图
            )
        except Exception as e:
            logger.warning(f"Failed to watch model with wandb: {str(e)}")
        
        
    def finish(self):
        """结束wandb运行"""
        if not self.enabled:
            return
        wandb.finish()