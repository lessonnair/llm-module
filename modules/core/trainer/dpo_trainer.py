# -*- coding: utf-8 -*-

from typing import Optional
import trl
from modules.util.custom_log import get_logger

logger = get_logger(__name__)


class DPOTrainer(trl.DPOTrainer):

    def __init__(self, **kwargs):
        trl.DPOTrainer.__init__(self, **kwargs)


    def save_model(self, output_dir: Optional[str] = None) -> None:
        if self.args.should_save:
            self._save(output_dir)