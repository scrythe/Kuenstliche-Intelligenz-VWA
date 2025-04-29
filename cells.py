import nbformat
import sys
import pyperclip

with open("Künstliche_Intelligenz.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

cells_text = []
start = int(sys.argv[1])
end = int(sys.argv[2])
selected_text: list[str] = []

for cell in nb.cells:
    # if cell.cell_type != "markdown":
    #     continue
    cells_text.append(cell.source)

for i in range(start,end+1):
    selected_text.append(cells_text[i])

selected_text.append("Vergiss nicht, nur notwendige Änderungen zu machen.")
text = "\n".join(selected_text)
print(text)
pyperclip.copy(text)
