from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from python.helpers import get_directory, check_path_exists
from .model_configs import TranslatorConfig


class Envit5Translation():
    config: TranslatorConfig
    src_lang: str | None = None
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

        self.device = torch.device("cpu")
        self.model.to(self.device)

    def set_tokenizer(self, src_code: str) -> None:
        self.src_lang = src_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_name)

    def inference(self, inputs: str = None, max_new_tokens: int = 500, num_beams: int = 1):
        inputs = [f"{self.src_lang}: {inputs}"]

        # inputs = [
        #     "vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam.",
        #     "vi: Theo báo cáo mới nhất của Linkedin về danh sách việc làm triển vọng với mức lương hấp dẫn năm 2020, các chức danh công việc liên quan đến AI như Chuyên gia AI (Artificial Intelligence Specialist), Kỹ sư ML (Machine Learning Engineer) đều xếp thứ hạng cao.",
        #     "en: Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing our research and tools to fuel progress in the field.",
        #     "en: We're on a journey to advance and democratize artificial intelligence through open source and open science."
        # ]

        generated_tokens = self.model.generate(self.tokenizer(
            inputs, return_tensors="pt", padding=True).input_ids.to(self.device), max_length=512)
        outputs = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)

        outputs = outputs[0][4:]
        print(outputs)

        return outputs
