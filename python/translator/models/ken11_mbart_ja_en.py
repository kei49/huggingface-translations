from transformers import (
    MBartForConditionalGeneration, MBartTokenizer
)


from python.helpers import get_directory, check_path_exists
from python.translator.model_configs import TranslatorConfig


class KEN11_MBART_JA_EN():
    config: TranslatorConfig
    src_lang: str | None = None
    to_lang: str | None = None
    tokenizer: MBartTokenizer
    model: MBartForConditionalGeneration

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config

        if not check_path_exists(config.model_path):
            self.model = MBartForConditionalGeneration.from_pretrained(
                config.pretrained_name)

            self.model.save_pretrained(config.model_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(
                config.model_path)

    def set_tokenizer(self) -> None:
        self.tokenizer = MBartTokenizer.from_pretrained(
            self.config.pretrained_name)

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
        inputs = self.tokenizer(inputs, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs, decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=512)
        outputs = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)[0]

        return outputs
