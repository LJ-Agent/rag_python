"""Proto compilation script. Run after editing .proto files."""
import subprocess
import sys
from pathlib import Path

PROTO_DIR = Path(__file__).parent.parent / "proto"
OUT_DIR = Path(__file__).parent.parent / "src" / "communication" / "grpc_server" / "generated"


def compile_protos():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for proto_file in PROTO_DIR.glob("*.proto"):
        print(f"Compiling {proto_file.name}...")
        subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{PROTO_DIR}",
                f"--python_out={OUT_DIR}",
                f"--grpc_python_out={OUT_DIR}",
                str(proto_file),
            ],
            check=True,
        )
    # Create __init__ for generated package
    (OUT_DIR / "__init__.py").touch()
    print(f"Done. Generated files in {OUT_DIR}")


if __name__ == "__main__":
    compile_protos()
