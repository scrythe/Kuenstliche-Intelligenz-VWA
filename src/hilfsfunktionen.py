from IPython.display import display, HTML
import base64

number = 1

zitierungen: list[str] = []


def erstelle_bild(bild, breite, label, name=None, year=None):
    global number
    base64_bild = base64.b64encode(bild.read()).decode()
    if name:
        zitierung = f"Abb. {number}: {label} ({name}, {year})"
    else:
        zitierung = f"Abb. {number}: {label} (Verf.)"

    zitierungs_id = f"abb{number}"
    zitierungen.append((zitierung, zitierungs_id))

    html = f"""
        <figure>
            <img style="width: {breite}px;" src="data:image/png;base64,{base64_bild}" alt="{label}">
            <figcaption id="{zitierungs_id}" >{zitierung}</figcaption>
        </figure>
        """
    display(HTML(html))
    number += 1
