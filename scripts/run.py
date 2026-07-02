import sys
import subprocess

if __name__ == "__main__":
    from pathlib import Path
    cmd = [sys.executable, str(Path(__file__).parent / "cli.py")] + sys.argv[1:]
    subprocess.run(cmd)
