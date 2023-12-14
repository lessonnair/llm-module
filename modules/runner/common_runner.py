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
        if hasattr(tokenizer, "padding_side"):
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

        self.generating_args = self.get_instance("generating_args")

        render = self.get_config("render")
        self.render = Render(render)

        self.model_path = self.get_config("pretrained_model_name_or_path")

        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()


    def _get_model(self):
        if self.model_path is not None and len(self.model_path) > 0:
            model = self.new_instance("model", model_path=self.model_path)
        else:
            model = self.get_instance("model")
        model.eval()
        model.requires_grad_(False)

        return model


    def _get_tokenizer(self):
        # if self.model_path is not None and len(self.model_path) > 0:
        #     tokenizer = self.new_instance("tokenizer", model_path=self.model_path)
        # else:
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

            # response = chat_model.chat(query, history)[0]
            # print(response)
            response = ''
            for new_text in chat_model.stream_chat(query, history):
                print(new_text, end="", flush=True)
                response += new_text

            history = history + [(query, response)]
















