# Huggingface Translators

## Setup python environments on MacOSX

```
pyenv local 3.10.6
poetry run pip3 install torch torchvision torchaudio sentencepiece protobuf==3.20.1
poetry update
source .venv/bin/activate
```

## Run python backend APIs

```
gunicorn main:app
```

## Run React UI app

```
yarn install
yarn start
```
