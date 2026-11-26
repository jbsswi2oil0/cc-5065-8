"""Launcher for the transient experiment window.

Run:  python3 -I run_server.py [--port 8000]
The graphical web UI is the primary entry point; the CLI is only a helper.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transient_renderer.server import serve

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    serve(a.host, a.port)
