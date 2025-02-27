## What I did (joe)

```sh
python -m pip venv .venv
source .venv/bin/activate

python -m pip install pip-tools
# < create requirements.in >
pip-compile requirements.in

python -m pip install -r requirements.txt
```

## Instructions for users

```sh
python -m pip venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
