from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from python.exceptions import InvalidLanguagesError
from python.helpers import get_directory, check_path_exists
from python.translator.model_configs import TranslatorConfig


class AutoTranslator():
    config: TranslatorConfig
    src_lang: str | None = None
    tgt_lang: str | None = None
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config

        if not check_path_exists(config.model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.pretrained_name)

            self.model.save_pretrained(config.model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_path)

    def set_languages(self, src_code: str = None, tgt_code: str = None) -> None:
        if src_code == None and tgt_code == None:
            pass

        if src_code != None:
            langs = [la for la in self.config.available_src_langs if src_code in la]
            if len(langs) == 0:
                raise InvalidLanguagesError(f"language {src_code=} is invalid")
            self.src_lang = langs[0]

        if tgt_code != None:
            langs = [la for la in self.config.available_tgt_langs if tgt_code in la]
            if len(langs) == 0:
                raise InvalidLanguagesError(f"language {tgt_code=} is invalid")
            self.tgt_lang = langs[0]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_name)

        if self.src_lang != None:
            self.tokenizer.src_lang = self.src_lang

        print("CURRENT: TOKENIZER: ", self.tokenizer)

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
        encoded = self.tokenizer(inputs, return_tensors="pt")
        forced_bos_token_id = self.tokenizer.lang_code_to_id[
            self.tgt_lang] if self.tgt_lang != None else None

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams)

        outputs = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)[0]

        return outputs
