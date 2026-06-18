import argparse  # REFACTOR
import subprocess  # REFACTOR
import sys  # REFACTOR
import time  # REFACTOR


def main():  # REFACTOR
    parser = argparse.ArgumentParser(description="VibAE-Monitor")  # REFACTOR
    parser.add_argument("--backend", action="store_true")  # REFACTOR
    parser.add_argument("--frontend", action="store_true")  # REFACTOR
    parser.add_argument("--all", action="store_true")  # REFACTOR
    args = parser.parse_args()  # REFACTOR

    if not (args.backend or args.frontend or args.all):  # REFACTOR
        parser.print_help()  # REFACTOR
        sys.exit(0)  # REFACTOR

    processes = []  # REFACTOR
    try:  # REFACTOR
        if args.backend or args.all:  # REFACTOR
            print("Backend  -> http://127.0.0.1:8000")  # REFACTOR
            processes.append(subprocess.Popen(  # REFACTOR
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]  # REFACTOR
            ))  # REFACTOR
            time.sleep(2)  # REFACTOR

        if args.frontend or args.all:  # REFACTOR
            print("Frontend -> http://127.0.0.1:8080/dashboard.html")  # REFACTOR
            processes.append(subprocess.Popen(  # REFACTOR
                [sys.executable, "-m", "http.server", "8080"], cwd="frontend"  # REFACTOR
            ))  # REFACTOR

        print("\nAppuyez sur Ctrl+C pour arreter.\n")  # REFACTOR
        while True:  # REFACTOR
            for p in processes:  # REFACTOR
                if p.poll() is not None:  # REFACTOR
                    raise KeyboardInterrupt  # REFACTOR
            time.sleep(1)  # REFACTOR

    except KeyboardInterrupt:  # REFACTOR
        print("\nArret des services...")  # REFACTOR
        for p in processes:  # REFACTOR
            try:  # REFACTOR
                p.terminate()  # REFACTOR
                p.wait(timeout=5)  # REFACTOR
            except Exception:  # REFACTOR
                p.kill()  # REFACTOR


if __name__ == "__main__":  # REFACTOR
    main()  # REFACTOR
