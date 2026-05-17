"""Standalone-TensorBoard-Viewer fuer das RL-Training.

In der IDE (VS Code / PyCharm / ...) einfach auf **Run** klicken --
startet einen TensorBoard-Server auf Port 6007, der das Logdir
`rl/tensorboard` liest, und oeffnet die URL im Standard-Browser.

Port 6007 bewusst gewaehlt, damit es nicht mit dem auto-gestarteten
TB-Server von `train.py` (Port 6006) kollidiert -- du kannst beide
parallel laufen lassen, z.B. um einen alten Run neben dem aktuellen
Training zu vergleichen.

Beenden: Strg+C im Run-Fenster der IDE, oder das Run-Panel stoppen.
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PORT = 6007


def main() -> int:
    script_dir = Path(__file__).resolve().parent         # .../plasma_cutter/rl
    project_root = script_dir.parent                      # .../plasma_cutter
    logdir = script_dir / "tensorboard"

    # venv-eigenes tensorboard bevorzugen, damit die Version zum
    # Training-Run passt und keine Event-File-Inkompatibilitaeten
    # auftreten.
    venv_tb = project_root / ".venv" / "Scripts" / "tensorboard.exe"
    if venv_tb.exists():
        tb_cmd: list[str] = [str(venv_tb)]
    else:
        # Fallback: tensorboard aus dem aktuellen Python-Interpreter
        # als Modul aufrufen. Funktioniert auch ohne venv.
        tb_cmd = [sys.executable, "-m", "tensorboard.main"]

    if not logdir.exists():
        print(f"[FEHLER] Logdir existiert nicht: {logdir}")
        print("Starte zuerst ein Training (train.py), damit Events erzeugt werden.")
        return 1

    url = f"http://localhost:{PORT}"
    print("=" * 57)
    print(" TensorBoard Viewer")
    print("=" * 57)
    print(f" Logdir : {logdir}")
    print(f" URL    : {url}")
    print(f" Port   : {PORT}")
    print("=" * 57)
    print()

    full_cmd = tb_cmd + [
        f"--logdir={logdir}",
        f"--port={PORT}",
        "--reload_multifile=true",
    ]
    print("Starte TensorBoard ...")
    print(" ".join(full_cmd))
    print()

    try:
        proc = subprocess.Popen(full_cmd)
    except FileNotFoundError as e:
        print(f"[FEHLER] TensorBoard konnte nicht gestartet werden: {e}")
        return 1

    # Kleine Wartezeit, damit der Server auch wirklich bindet, bevor
    # der Browser die URL aufruft -- sonst sieht man einmalig einen
    # "Seite nicht erreichbar"-Ladeversuch.
    time.sleep(2.0)
    print(f"Oeffne Browser: {url}")
    webbrowser.open(url)
    print()
    print("TensorBoard laeuft. Beenden mit Strg+C oder Stop-Button in der IDE.")
    print()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nBeende TensorBoard ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
