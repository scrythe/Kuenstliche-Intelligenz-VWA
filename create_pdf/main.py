from playwright.sync_api import sync_playwright
import pymupdf  # type: ignore
from pikepdf import Pdf, OutlineItem
import os


def extract_keywords(file, keywords):
    results = []

    current_keyword_index = 0
    doc = pymupdf.open(file)
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


def merge_pdf_cover(file, export):
    pdf = Pdf.open(file)
    src = Pdf.open("VWA-Titelblatt.pdf")
    del pdf.pages[0]
    pdf.pages.insert(0, src.pages[0])
    pdf.save(export)


def merge_pdf_unterschrift(file, export):
    pdf = Pdf.open(file)
    src = Pdf.open("Selbstständigkeitserklärung.pdf")
    pdf.pages.extend(src.pages)
    pdf.save(export)


def add_bookmarks(file, headers):
    doc = pymupdf.open(file)
    page = doc.load_page(0)
    page_height = page.rect.height
    doc.close()

    pdf = Pdf.open(file)
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
        browser = playwright.chromium.launch()
        page = browser.new_page()
        ki_file = os.path.join(os.getcwd(), "Künstliche_Intelligenz.html")
        page.goto(ki_file, wait_until="networkidle")

        abb_keywords = page.evaluate("abbKeywords")
        header_keywords = page.evaluate("headerKeywords")
        page.pdf(path="temp.pdf", format="A4")
        merge_pdf_cover("temp.pdf", "temp_cover.pdf")
        abbs = extract_keywords("temp_cover.pdf", abb_keywords)
        headers = extract_keywords("temp_cover.pdf", header_keywords)
        page.evaluate("generateAbb", abbs)
        page.evaluate("generateTOC", headers)
        page.pdf(path="temp.pdf", format="A4")
        browser.close()
        merge_pdf_cover("temp.pdf", "temp_cover.pdf")
        merge_pdf_unterschrift("temp_cover.pdf", "temp_unterschrift.pdf")

    add_bookmarks("temp_unterschrift.pdf", headers)

    os.remove("temp.pdf")
    os.remove("temp_cover.pdf")
    os.remove("temp_unterschrift.pdf")
