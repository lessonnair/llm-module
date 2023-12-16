# -*- coding: utf-8 -*-

import tiktoken
from modules.util.custom_log import get_logger
from modules.util.config_util import render_config
from typing import Dict, List, Optional, Tuple, Union

logger = get_logger(__name__)


class Render(object):

    def __init__(self, name):
        config = render_config.get_section_kvs(name)

        self.prefix = config.get("prefix")
        self.prompt = config.get("prompt")
        self.system = config.get("system")
        self.sep = config.get("sep")
        self.stop_words = config.get("stop_words")
        self.efficient_eos = config.get("efficient_eos")
        self.use_history = config.get("use_history", True)

    def xijin_get_special_ids(self,
                         tokenizer: "PreTrainedTokenizer"
                         ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else:
            # for baichuan, qwen and gpt2 models
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        # for baichuan, qwen, chatglm
        if self.efficient_eos:
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]
        return bos_ids, eos_ids

    def render_with_history(self,
                            tokenizer: "PreTrainedTokenizer",
                            query: str,
                            resp: str,
                            history: Optional[List[Tuple[str, str]]] = None,
                            system: Optional[str] = None,
                            multi_turn: bool = False
                            ) -> Tuple[List[int], List[int]]:
        system = system or self.system
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]

        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self.render(tokenizer, context=self.sep)
        pairs = []

        for idx, (query, resp) in enumerate(history):
            if idx == 0:
                prefix_ids = self.render(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) > 0:
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self.render(tokenizer, context=self.prompt, query=query, idx=str(idx))
            resp_ids = self.render(tokenizer, context=resp)
            pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))

        if multi_turn:
            return pairs

        prompt_ids = []
        for query_ids, resp_ids in pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids += pairs[-1][0]
        answer_ids = pairs[-1][1]
        return prompt_ids, answer_ids

    def render(self,
               tokenizer: "PreTrainedTokenizer",
               context: List[Union[str, Dict[str, str]]],
               system: Optional[str] = None,
               query: Optional[str] = None,
               idx: Optional[str] = None
               ) -> List[int]:
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding):
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for c in context:
            if isinstance(c, str):
                for tag in ["system", "query", "idx"]:
                    c = c.replace("{{" + tag + "}}", locals()[tag], 1) if locals()[tag] is not None else c
                if len(c) > 0:
                    token_ids += tokenizer.encode(c, **kwargs)
            elif isinstance(c, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(c.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(c)))
        return token_ids


if __name__ == '__main__':
    render = Render("chatglm2")
    pass
