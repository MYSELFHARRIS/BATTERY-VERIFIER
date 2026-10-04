import argparse
import getpass
import sys
from pathlib import Path

# Add project root to sys.path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web.auth import create_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a user in the SQLite store.")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument(
        "--db",
        default="outputs/users.db",
        help="Path to the users SQLite database (default: outputs/users.db)",
    )
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Confirm password: ")

    if password != confirm_password:
        print("Error: Passwords do not match.")
        sys.exit(1)

    try:
        create_user(args.db, args.email, password)
        print("User created")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
