import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from python.models.mbert_large_50 import MBartLargeManyToMany
from python.exceptions import InvalidLanguagesError
from .model_configs import ModelType, get_translator_config
from .auto_translator import AutoTranslator
from .envit5_translation import Envit5Translation
from .k024_mt5_zh_ja_en_trimmed import K024_MT5_ZH_JA_EN_TRIMMED
from .ken11_mbart_ja_en import KEN11_MBART_JA_EN


class Translator():
    supported_la = ["en", "es", "fr", "it", "ja",
                    "ko", "ru", "vi", "zh", "id", "pl", "th"]
    use_gpu = torch.cuda.is_available()

    def __init__(self, model_type: ModelType) -> None:
        self.model_config = get_translator_config(model_type)
        self.model_type = model_type
        self._load_model()

    def _load_model(self) -> None:
        print("loading model")
        if self.model_type == ModelType.MBART_LARGE_MANY_TO_MANY and self.use_gpu:
            self.model = MBartLargeManyToMany(use_gpu=self.use_gpu)
        elif self.model_type == ModelType.ENVIT5_TRANSLATION:
            self.model = Envit5Translation(self.model_config)
        elif self.model_type == ModelType.K024_MT5_ZH_JA_EN_TRIMMED:
            self.model = K024_MT5_ZH_JA_EN_TRIMMED(self.model_config)
        elif self.model_type == ModelType.KEN11_MBART_JA_EN:
            self.model = KEN11_MBART_JA_EN(self.model_config)
        else:
            self.model = AutoTranslator(self.model_config)

    def set_languages(self, src_code: str, tgt_code: str) -> None:
        if self.model_type == ModelType.ENVIT5_TRANSLATION:
            self.model.set_tokenizer(src_code)
            return
        elif self.model_type == ModelType.KEN11_MBART_JA_EN:
            self.model.set_tokenizer()
            return

        if self.model and src_code in self.supported_la and tgt_code in self.supported_la:
            if len(self.model.config.available_src_langs) == 1:
                src_code = None

            if len(self.model.config.available_tgt_langs) == 1:
                tgt_code = None

            self.model.set_languages(src_code, tgt_code)
        else:
            raise InvalidLanguagesError("Failed to set_languages")

    def inference(self, texts) -> str:
        if self.model:
            print("start to inference from: ", texts)
            outputs = self.model.inference(texts)
            print("get outputs: ", outputs)
            return outputs
        return
