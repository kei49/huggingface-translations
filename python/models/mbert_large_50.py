from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..helpers import get_directory, check_path_exists

NAME = "mbart-large-50-many-to-many-mmt"


class MBartLargeManyToMany():
    originally_available_locales = [
        "ar_AR", "cs_CZ" "de_DE", "en_XX", "es_XX" "et_EE", "fi_FI", "fr_XX",
        "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV",
        "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU" "si_LK", "tr_TR", "vi_VN",
        "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID",
        "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
        "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA",
        "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

    supported_locales = ["en_XX", "es_XX", "fr_XX", "it_IT", "ja_XX",
                         "ko_KR", "ru_RU", "vi_VN", "zh_CN", "id_ID", "pl_PL", "th_TH"]

    def __init__(self, use_gpu: bool = False) -> None:
        name = NAME
        directory = get_directory(name)
        self.pretrined_name = f"facebook/{name}"
        self.use_gpu = use_gpu

        if use_gpu:
            print(f"using gpu for {NAME}")

        if not check_path_exists(directory):
            if use_gpu:
                self.model = MBartForConditionalGeneration.from_pretrained(
                    self.pretrined_name)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.pretrined_name)

            self.model.save_pretrained(directory)
        else:
            if use_gpu:
                self.model = MBartForConditionalGeneration.from_pretrained(
                    directory)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    directory)

    def set_languages(self, from_la: str, to_la: str) -> None:
        use_locales = []
        for la in [from_la, to_la]:
            locales = [
                locale for locale in self.supported_locales if la in locale]
            if len(locales) == 0:
                raise Exception(f"language {la=} is not supported")

            use_locales.append(locales[0])

        [src_lang, tgt_lang] = use_locales

        if self.use_gpu:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                self.pretrined_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrined_name)

        self.tokenizer.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
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


"""
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

MBartLargeBase is created to fine-tune models for translation tasks
"""

# NAME = "mbart-large-50"


class MBartLargeBase():
    originally_available_locales = [
        "ar_AR", "cs_CZ" "de_DE", "en_XX", "es_XX" "et_EE", "fi_FI", "fr_XX",
        "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV",
        "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU" "si_LK", "tr_TR", "vi_VN",
        "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID",
        "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
        "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA",
        "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

    supported_locales = ["en_XX", "es_XX", "fr_XX", "it_IT", "ja_XX",
                         "ko_KR", "ru_RU", "vi_VN", "zh_CN", "id_ID", "pl_PL", "th_TH"]

    def __init__(self, from_la: str, to_la: str) -> None:
        use_locales = []
        for la in [from_la, to_la]:
            locales = [
                locale for locale in self.supported_locales if la in locale]
            if len(locales) == 0:
                raise Exception(f"language {la=} is not supported")

            use_locales.append(locales[0])

        name = NAME
        pretrined_name = f"facebook/{name}"
        directory = get_directory(name)

        [src_lang, tgt_lang] = use_locales

        print(f"{pretrined_name=}, {src_lang=}, {tgt_lang=}")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            pretrined_name, src_lang=src_lang, tgt_lang=tgt_lang)

        if not check_path_exists(directory):
            self.model = MBartForConditionalGeneration.from_pretrained(
                pretrined_name)
            self.model.save_pretrained(directory)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(
                directory)

    def inference(self, inputs: str = None):
        inputs = self.tokenizer(inputs, return_tensors="pt")
        generated_ids = self.model.generate(
            inputs["input_ids"], num_beams=4, max_length=5)
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]

        return outputs
