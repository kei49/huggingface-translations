from dataclasses import dataclass
from typing import Dict

from .translator import Translator, ModelType


@dataclass
class TranslateParams:
    model_type: ModelType
    from_la: str
    to_la: str


class Manager():
    loaded_models: Dict[str, Translator] = {}

    initial_models = [
        # TranslateParams(ModelType.OPUS_MT_KO_EN, "ko", "en"),
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

        print(f"Model: {model_id=}")

        return self.loaded_models[model_id]

    def _load_model(self, p: TranslateParams) -> Translator:
        model = Translator(p.model_type)
        model.set_languages(p.from_la, p.to_la)

        model_id = self._get_model_id(p)
        self._set_model(model_id, model)

        print(f"Loaded {model_id=}")

    def _set_model(self, model_id, model):
        self.loaded_models[model_id] = model

    def _get_model_id(self, p: TranslateParams) -> str:
        return f"{p.model_type}_{p.from_la}_{p.to_la}"
