# Huggingface Translators

## Setup python environments on MacOSX

```
pyenv local 3.10.6
poetry update
poetry run pip3 install torch torchvision torchaudio sentencepiece protobuf==3.20.1 sacremoses
source .venv/bin/activate
```

## Run python backend APIs

```
gunicorn main:app
```
