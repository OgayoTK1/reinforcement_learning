"""AgriResilience RL entry point.

    uv run main.py                 # start the viewer at http://127.0.0.1:8000 (opens a browser)
    uv run main.py serve --port 9000 --no-browser
    uv run main.py train  [...]    # run the experiment matrix  (see training/experiment_runner.py)
    uv run main.py evaluate        # held-out test and out-of-distribution evaluation
    uv run main.py export          # copy the best checkpoints into models/
    uv run main.py report          # figures, tables and the report from experiments/results
    uv run main.py play  [...]     # headless playback of a checkpoint in the terminal (play.py)
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def serve(host: str, port: int, browser: bool) -> None:
    import uvicorn
    if browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv and not argv[0].startswith("-") else "serve"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if cmd == "serve":
        ap = argparse.ArgumentParser(prog="main.py serve")
        ap.add_argument("--host", default="127.0.0.1")
        ap.add_argument("--port", type=int, default=8000)
        ap.add_argument("--no-browser", action="store_true")
        a = ap.parse_args(rest)
        serve(a.host, a.port, not a.no_browser)
    elif cmd == "train":
        from training.experiment_runner import main as run
        sys.argv = ["experiment_runner"] + rest
        run()
    elif cmd == "export":
        from training.export_models import main as run
        sys.argv = ["export_models"] + rest
        run()
    elif cmd == "evaluate":
        from evaluation.final_eval import main as run
        sys.argv = ["final_eval"] + rest
        run()
    elif cmd == "report":
        from reporting.build_all import main as run
        sys.argv = ["build_all"] + rest
        run()
    elif cmd == "play":
        import play
        sys.argv = ["play"] + rest
        play.main()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
