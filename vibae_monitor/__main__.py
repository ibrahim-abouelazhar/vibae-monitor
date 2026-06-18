import sys
import os

# Add root folder to sys.path to find run.py
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from run import main

if __name__ == "__main__":
    main()
