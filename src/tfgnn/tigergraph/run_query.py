import argparse
import json

from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one installed TigerGraph query")
    parser.add_argument("query_name")
    parser.add_argument(
        "--params-json", default="{}", help="JSON object of installed-query parameters"
    )
    args = parser.parse_args()
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise ValueError("--params-json must decode to an object")
    settings = Settings()
    result = Client(settings).run_installed_with_timeout(
        args.query_name, params, timeout_s=settings.query_timeout_s
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
