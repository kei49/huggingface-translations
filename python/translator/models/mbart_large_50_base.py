from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from python.helpers import get_directory, check_path_exists
from python.translator.model_configs import TranslatorConfig


class MBartLarge50Base():
    config: TranslatorConfig
    src_lang: str | None = None
    to_lang: str | None = None
    tokenizer: AutoTokenizer | MBart50TokenizerFast
    model: AutoModelForSeq2SeqLM | MBartForConditionalGeneration

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config

        if config.use_gpu:
            print(f"using gpu for {config.pretrained_name}")

        if not check_path_exists(config.model_path):
            if config.use_gpu:
                self.model = MBartForConditionalGeneration.from_pretrained(
                    config.pretrained_name)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.pretrained_name)

            self.model.save_pretrained(config.model_path)
        else:
            if config.use_gpu:
                self.model = MBartForConditionalGeneration.from_pretrained(
                    config.model_path)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.model_path)

    def set_languages(self, src_lang: str, tgt_lang: str) -> None:
        print(f"set_languages: {src_lang=}, {tgt_lang=}")

        if self.config.use_gpu:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                self.config.pretrained_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.pretrained_name)

        self.tokenizer.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
        print("inferencing with", inputs, self.tokenizer, self.tgt_lang)

        encoded = self.tokenizer(inputs, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams)

        outputs = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)[0]

        print(outputs)

        return outputs
