from typing import Dict

from .translator import Translator, ModelType
from .model_configs import TranslateParams
from python.logger import use_logger
logger = use_logger(__name__)


class Manager():
    loaded_models: Dict[str, Translator] = {}

    initial_models = [
        # TranslateParams(ModelType.OPUS_MT_KO_EN, "ko", "en"),
        # TranslateParams(ModelType.ENVIT5_TRANSLATION, "en", "vi"),
        # TranslateParams(ModelType.ENVIT5_TRANSLATION, "vi", "en"),
        # TranslateParams(ModelType.OPUS_MT_MUL_EN, "ja", "en"),
        # TranslateParams(ModelType.OPUS_MT_MUL_EN, "vi", "en"),
        # TranslateParams(ModelType.MT5_SMALL, "en", "ja")
        # TranslateParams(ModelType.OPUS_MT_MUL_EN, "zh", "en"),
        # TranslateParams(ModelType.OPUS_MT_MUL_EN, "id", "en"),
        # TranslateParams(ModelType.OPUS_MT_MUL_EN, "th", "en"),
        # TranslateParams(ModelType.MBART_LARGE_MANY_TO_MANY, "en", "ja"),
        # TranslateParams(ModelType.MBART_LARGE_MANY_TO_MANY, "en", "ko"),
        # TranslateParams(ModelType.MBART_LARGE_MANY_TO_MANY, "en", "vi"),
    ]

    def __init__(self) -> None:
        for model in self.initial_models:
            self._load_model(model)

    def get_model(self, p: TranslateParams) -> Translator:
        model_id = self._get_model_id(p)

        if not model_id in self.loaded_models:
            self._load_model(p)

        logger.info(f"Model: {model_id=}")

        return self.loaded_models[model_id]

    def _load_model(self, p: TranslateParams) -> Translator:
        model = Translator(p.model_type)
        model.set_languages((p.src_lang if p.src_lang else p.from_la),
                            (p.tgt_lang if p.tgt_lang else p.to_la))

        model_id = self._get_model_id(p)
        self._set_model(model_id, model)

        logger.info(f"Newly loaded {model_id=}")

    def _set_model(self, model_id, model):
        self.loaded_models[model_id] = model

    def _get_model_id(self, p: TranslateParams) -> str:
        return f"{p.model_type}_{p.from_la}_{p.to_la}"
