import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="VibAE-Monitor")
    parser.add_argument("--backend", action="store_true")
    parser.add_argument("--frontend", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not (args.backend or args.frontend or args.all):
        parser.print_help()
        sys.exit(0)

    processes = []
    try:
        if args.backend or args.all:
            print("Backend  -> http://127.0.0.1:8000")
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
            ))
            time.sleep(2)

        if args.frontend or args.all:
            print("Frontend -> http://127.0.0.1:8080/dashboard.html")
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "http.server", "8080"], cwd="frontend"
            ))

        print("\nAppuyez sur Ctrl+C pour arreter.\n")
        while True:
            for p in processes:
                if p.poll() is not None:
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nArret des services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
