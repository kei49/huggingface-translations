from python.helpers import get_directory, check_path_exists
from .model_configs import TranslatorConfig


from python.helpers import check_path_exists, get_directory
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    Text2TextGenerationPipeline,
)


class K024_MT5_ZH_JA_EN_TRIMMED():
    config: TranslatorConfig
    src_lang: str | None = None
    to_lang: str | None = None
    tokenizer: T5Tokenizer
    model: MT5ForConditionalGeneration

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config

        if not check_path_exists(config.model_path):
            self.model = MT5ForConditionalGeneration.from_pretrained(
                config.pretrained_name)

            self.model.save_pretrained(config.model_path)
        else:
            self.model = MT5ForConditionalGeneration.from_pretrained(
                config.model_path)

    def set_languages(self, src_code: str, tgt_code) -> None:
        self.src_lang = src_code
        self.to_lang = tgt_code

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.pretrained_name)

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
        sentence = f"{self.src_lang}2{self.to_lang}: {inputs}"

        pipe = Text2TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer
        )

        res = pipe(sentence, max_length=512, num_beams=num_beams)
        outputs = res[0]['generated_text']

        print(outputs)

        return outputs
