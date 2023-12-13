# -*- coding: utf-8 -*-

import torch
from threading import Thread
from typing import Any, Dict, Generator, List, Optional, Tuple
from transformers import GenerationConfig, TextIteratorStreamer
from modules.core.render.base_render import Render
from modules.util.metric_util import get_logits_processor


class ChatModel:

    def __init__(self,
                 model,
                 tokenizer,
                 generating_args,
                 render: Render
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.generating_args = self._init_generating_args(generating_args)
        self.render = render

    def _init_generating_args(self, generating_args):
        if generating_args.get("pad_token_id") is None:
            generating_args["pad_token_id"] = self.tokenizer.pad_token_id
        if generating_args.get("eos_token_id") is None:
            generating_args["eos_token_id"] = self.tokenizer.eos_token_id
        return generating_args

    def preprocess(self,
                   query: str,
                   history: Optional[List[Tuple[str, str]]] = None,
                   system: Optional[str] = None,
                   **input_kwargs
                   ) -> Tuple[Dict[str, Any], int]:

        prompt, _ = self.render.render_with_history(
            tokenizer=self.tokenizer,
            query=query, resp="", history=history, system=system,
            multi_turn=False
        )
        device = None
        if hasattr(self.model, "device"):
            device = self.model.device
        input_ids = torch.tensor([prompt], device=device)
        kwargs = dict(
            inputs=input_ids,
            # generation_config=GenerationConfig(**self.generating_args),
            logits_processor=get_logits_processor()
        )
        prompt_length = len(input_ids[0])
        return kwargs, prompt_length

    def chat(self,
             query: str,
             history: Optional[List[Tuple[str, str]]] = None,
             system: Optional[str] = None,
             **input_kwargs
             ) -> Tuple[str, Tuple[int, int]]:
        kwargs, prompt_length = self.preprocess(query, history, system, **input_kwargs)
        generation_output = self.model.generate(**kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    @torch.inference_mode()
    def stream_chat(self,
                    query: str,
                    history: Optional[List[Tuple[str, str]]] = None,
                    system: Optional[str] = None,
                    **input_kwargs
                    ) -> Generator[str, None, None]:
        kwargs, _ = self.preprocess(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=60.0,
                                        skip_prompt=True,
                                        skip_special_tokens=True
                                        )
        kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        yield from streamer

