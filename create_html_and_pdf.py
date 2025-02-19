import subprocess
import sys
import create_pdf
import create_pdf.main

subprocess.run(
    [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--template",
        "./scrythe.tpl",
        "./Künstliche_Intelligenz.ipynb",
    ],
)

create_pdf.main.create_pdf()
