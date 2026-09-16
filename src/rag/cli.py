import json

import typer

from .config import Settings
from .index import ingest as build_index

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest():
    """Replace the index snapshot from MD/TXT documents."""
    typer.echo(f"Indexed chunks: {build_index(Settings())}")


@app.command()
def ask(question: str):
    """Ask an independent question."""
    from .runtime import answer
    result = answer(question)
    typer.echo(json.dumps({k: result[k] for k in
                          ("answer", "citations", "is_answered", "trace")},
                         ensure_ascii=False, indent=2))
    if not result["is_answered"]:
        raise typer.Exit(2)


@app.command()
def serve():
    """Serve on loopback; not intended for direct public access."""
    import uvicorn
    uvicorn.run("rag.api:app", host="127.0.0.1", port=8000)
