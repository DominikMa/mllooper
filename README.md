# ML Looper

The documentation will be [here](https://mllooper.readthedocs.io).

## Installation with uv

Development uses Python 3.14; the library supports Python 3.10 and newer.

```bash
uv sync                             # GPU, with dev tools
uv sync --no-group gpu --group cpu  # CPU only, with dev tools
uv sync --no-dev                    # GPU, without dev tools
```

For CPU mode, also pass `--no-group gpu --group cpu` to `uv run`.

Run checks with `bash scripts/check`. Enable the pre-push hook with
`git config core.hooksPath scripts/hooks`.
