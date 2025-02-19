from playwright.sync_api import sync_playwright
import fitz  # type: ignore
from pikepdf import Pdf, OutlineItem
import os


def extract_headers(header_keywords):
    headers = []
    doc = fitz.open("temp.pdf")

    for page_num in range(doc.page_count):  # Use `doc.page_count` directly
        page = doc.load_page(page_num)
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if block["type"] == 0  # Text block
            for line in block["lines"]
            for span in line["spans"]
        ]

        headers.extend(
            {
                "text": span["text"].strip(),
                "page": page_num + 1,
                "bbox": span["bbox"],
            }
            for span in spans
            if any(header in span["text"] for header in header_keywords)
        )
    doc.close()
    return headers


def add_bookmarks(headers):
    doc = fitz.open("temp.pdf")
    page = doc.load_page(0)
    page_height = page.rect[2]
    doc.close()

    pdf = Pdf.open("temp.pdf")
    with pdf.open_outline() as outline:
        outline_hirachy = {}
        for header in headers:
            text = header["text"]
            level = text.count(".")
            page = header["page"]
            y = header["bbox"][1]

            # page_height = page.mediabox[3]  # Get the height of the page
            oi = OutlineItem(text, page - 1, "XYZ", top=page_height - y, zoom=1.0)
            outline_hirachy[level] = oi
            if level == 0:
                outline.root.append(oi)
            else:
                parent = outline_hirachy[level - 1]
                parent.children.append(oi)

    output_pdf_path = "Künstliche_Intelligenz.pdf"
    pdf.save(output_pdf_path)
    pdf.close()


def create_pdf():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        ki_file = os.path.join(os.getcwd(), "Künstliche_Intelligenz.html")
        page.goto(ki_file, wait_until="networkidle")

        abb_keywords = page.evaluate("abbKeywords")
        header_keywords = page.evaluate("headerKeywords")
        page.pdf(path="temp.pdf", format="A4")
        abbs = extract_headers(abb_keywords)
        headers = extract_headers(header_keywords)
        page.evaluate("generateAbb", abbs)
        page.evaluate("generateTOC", headers)
        page.pdf(path="temp.pdf", format="A4")
        browser.close()

    add_bookmarks(headers)
    os.remove("temp.pdf")


# create_pdf()
