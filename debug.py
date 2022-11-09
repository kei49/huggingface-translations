from transformers import (
    MBartForConditionalGeneration, MBartTokenizer
)
from python.helpers import check_path_exists, get_directory
# from transformers import (
#     T5Tokenizer,
#     MT5ForConditionalGeneration,
#     Text2TextGenerationPipeline,
# )
# path = "K024/mt5-zh-ja-en-trimmed"
name = "ken11/mbart-ja-en"
path = get_directory(name.replace('/', '_'))

# if not check_path_exists(name):
#     model = MT5ForConditionalGeneration.from_pretrained(path)

#     model.save_pretrained(name)
# else:
#     model = MT5ForConditionalGeneration.from_pretrained(name)


# pipe = Text2TextGenerationPipeline(
#     model=model,   # MT5ForConditionalGeneration.from_pretrained(path),
#     tokenizer=T5Tokenizer.from_pretrained(path),
# )

# sentence = "ja2zh: 吾輩は猫である。名前はまだ無い。"
# res = pipe(sentence, max_length=100, num_beams=4)
# res[0]['generated_text']


# print(res[0]['generated_text'])


if not check_path_exists(path):
    model = MBartForConditionalGeneration.from_pretrained(name)

    model.save_pretrained(path)
else:
    model = MBartForConditionalGeneration.from_pretrained(path)

tokenizer = MBartTokenizer.from_pretrained(name)

inputs = tokenizer("日本語の入力が長い場合には、どのような結果が得られるでしょうか", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=48)
pred = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(pred)
