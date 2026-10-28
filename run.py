"""
Main Application Launcher - AI Face ID Security System
Lead AI ML Engineering - Dual Engine (XGBoost + Random Forest)
"""

import sys
import tkinter as tk
from app_gui import FaceIDSecurityApp

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import cli
        sys.argv.pop(1)
        cli.main()
    else:
        root = tk.Tk()
        app = FaceIDSecurityApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
