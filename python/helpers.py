import os


def get_directory(name: str):
    return f"{os.getcwd()}/models/{name}"


def check_path_exists(path):
    return os.path.exists(path)


def create_flan_t5_langauges_texts():
    input = "English, Spanish, Japanese, Persian, Hindi, French, Chinese, Bengali, Gujarati, German, Telugu, Italian, Arabic, Polish, Tamil, Marathi, Malayalam, Oriya, Panjabi, Portuguese, Urdu, Galician, Hebrew, Korean, Catalan, Thai, Dutch, Indonesian, Vietnamese, Bulgarian, Filipino, Central Khmer, Lao, Turkish, Russian, Croatian, Swedish, Yoruba, Kurdish, Burmese, Malay, Czech, Finnish, Somali, Tagalog, Swahili, Sinhala, Kannada, Zhuang, Igbo, Xhosa, Romanian, Haitian, Estonian, Slovak, Lithuanian, Greek, Nepali, Assamese, Norwegian"
    output = input.replace(' ', '').split(',')

    texts = ""

    for o in output:
        texts = texts + '"' + o + '", '

    print(texts)
