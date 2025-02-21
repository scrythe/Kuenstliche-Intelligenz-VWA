from playwright.sync_api import sync_playwright
import pymupdf  # type: ignore
from pikepdf import Pdf, OutlineItem
import os


def extract_keywords(keywords):
    results = []

    current_keyword_index = 0
    doc = pymupdf.open("temp.pdf")
    for page in doc:
        while True:
            if current_keyword_index >= len(keywords):
                break
            current_keyword = keywords[current_keyword_index]
            rect = page.search_for(current_keyword)
            if len(rect):
                current_keyword_index += 1
                result = (current_keyword, page.number + 1, rect[0])
                results.append(result)
                continue
            break
    doc.close()
    return results


def add_bookmarks(headers):
    doc = pymupdf.open("temp.pdf")
    page = doc.load_page(0)
    page_height = page.rect.height
    doc.close()

    pdf = Pdf.open("temp.pdf")
    with pdf.open_outline() as outline:
        outline_hirachy = {}
        for header in headers:
            text = header[0]
            level = text.count(".")
            page = header[1]
            rect = header[2]

            oi = OutlineItem(text, page - 1, "XYZ", top=page_height - rect.y0, zoom=1.0)
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
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        ki_file = os.path.join(os.getcwd(), "Künstliche_Intelligenz.html")
        page.goto(ki_file, wait_until="networkidle")

        abb_keywords = page.evaluate("abbKeywords")
        header_keywords = page.evaluate("headerKeywords")
        page.pdf(path="temp.pdf", format="A4")
        abbs = extract_keywords(abb_keywords)
        headers = extract_keywords(header_keywords)
        page.evaluate("generateAbb", abbs)
        page.evaluate("generateTOC", headers)
        page.pdf(path="temp.pdf", format="A4")
        browser.close()

    add_bookmarks(headers)
    os.remove("temp.pdf")
