import argparse
import sys
from pathlib import Path

# Add project root to sys.path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Battery Verifier web application.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
