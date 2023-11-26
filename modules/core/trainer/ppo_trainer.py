
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from transformers import GenerationConfig, Trainer, TrainerState, TrainerControl
import trl
from modules.util.custom_log import get_logger
from modules.util.analyze_util import count_parameters

logger = get_logger(__name__)

class PPOTrainer(trl.PPOTrainer, Trainer):

    def __init__(self,
                 training_args: "Seq2SeqTrainingArguments",
                 callbacks: List["TrainerCallback"],
                 **kwargs):
        trl.PPOTrainer.__init__(self, **kwargs)
        if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
            raise ValueError("PPOTrainer is incompatible with DeepSpeed.")

        self.args = training_args

        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]


    def ppo_train(self):
        total_train_batch_size = (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {count_parameters(self.model)[0]}")

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)







