from IPython.display import display, HTML
import base64

number = 1


def erstelle_bild(bild, breite, label, name=None, year=None):
    global number
    base64_bild = base64.b64encode(bild.getvalue()).decode()
    if name:
        zitierung = f"Abb. {number}: {label} ({name}, {year})"
    else:
        zitierung = f"Abb. {number}: {label} (Verf.)"
    html = f"""
        <figure style="display: flex; flex-flow: column;">
            <img style="width: {breite}px;" src="data:image/png;base64,{base64_bild}" alt="{label}">
            <figcaption">{zitierung}</figcaption>
        </figure>
        """
    display(HTML(html))
    number += 1
