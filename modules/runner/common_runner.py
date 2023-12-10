# -*- coding: utf-8 -*-

from .basic_runner import Task
from modules.core.render.base_render import Render
from modules.core.chat.stream_chat import ChatModel

class Export(Task):
    def __init__(self, config, name=None):
        super(Export, self).__init__(config, name=name)

        self.output_dir = self.get_config("output_dir")
        self.max_shard_size = self.get_config("max_shard_size")

    def main_handle(self):
        model = self.get_instance("model")
        tokenizer = self.get_instance("TokenizerLoader")

        model.config.use_cache = True
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"
        model.save_pretrained(self.output_dir, max_shard_size=self.max_shard_size)
        try:
            tokenizer.save_pretrained(self.output_dir)
        except:
            self.logger.warning("cannot save tokenizer, please copy the files manually.")


class Chat(Task):

    def __init__(self, config, name=None):
        super(Chat, self).__init__(config, name=name)


        self.model = self.get_instance("model")
        self.generating_args = self.get_instance("generating_args")

        render = self.get_config("render")
        self.render = Render(render)

        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        tokenizer = self.get_instance("tokenizer")
        tokenizer.padding_side = "left"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "<|endoftext|>"
            self.logger.info("Add eos token: {}".format(tokenizer.eos_token))

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info("Add pad token: {}".format(tokenizer.pad_token))

        if self.render.stop_words is not None:
            tokenizer.add_special_tokens(
                dict(additional_special_tokens=self.render.stop_words),
                replace_additional_special_tokens=False
            )
        return tokenizer

    def main_handle(self):

        chat_model = ChatModel(self.model,
                               self.tokenizer,
                               self.generating_args,
                               self.render)

        history = []
        self.logger.info("******* Chat Demo *********")

        while True:
            try:
                query = input("\nUser:")
            except UnicodeDecodeError as e:
                self.logger.warning("input decoding error", e)
                continue
            except Exception:
                raise

            if query.strip() in ["exit", "quit"]:
                break

            if query.strip() == "clear":
                history = []
                self.logger("History has been removed.")
                continue

            print("Assistant: ", end="", flush=True)

            response = ""
            for new_text in chat_model.stream_chat(query, history):
                print(new_text, end="", flush=True)
                response += new_text

            history = history + [(query, response)]
















