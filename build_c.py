"""Build with MSYS2 GCC on Windows, or GCC on Linux."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parent
    name = "simulation.dll" if sys.platform == "win32" else "libsimulation.so"
    output = root / "csrc" / name
    command = [os.environ.get("CC", "gcc"), "-std=c99", "-O3", "-Wall",
               "-Wextra", "-Wpedantic", "-shared"]
    if sys.platform != "win32":
        command.append("-fPIC")
    # A failed compilation must not replace a previously working library.
    with tempfile.TemporaryDirectory(dir=root / "csrc") as temporary:
        built = Path(temporary) / name
        command += ["csrc/simulation.c", "csrc/physics.c", "-o", str(built), "-lm"]
        try:
            subprocess.run(command, cwd=root, check=True)
            built.replace(output)
            if sys.platform == "win32":
                # Moving from TemporaryDirectory preserves its private ACL.
                # Reset against csrc so the interactive user can load the DLL.
                subprocess.run(["icacls", str(output), "/reset"],
                               check=True, stdout=subprocess.DEVNULL)
        except FileNotFoundError as error:
            raise SystemExit("GCC not found. Put MSYS2 UCRT64 GCC on PATH or set CC.") from error
        except PermissionError as error:
            raise SystemExit("Cannot replace the library. Stop Streamlit and other Python processes using it, then rebuild.") from error
    print(output)


if __name__ == "__main__":
    main()
