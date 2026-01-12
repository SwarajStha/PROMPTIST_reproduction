import markdown
import os
from xhtml2pdf import pisa

INPUT_MD = "PROJECT_REPORT.md"
OUTPUT_PDF = "Promptist_Report.pdf"
OUTPUT_HTML = "Promptist_Report.html"

# CSS for styling the PDF and HTML
CSS = """
<style>
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: Helvetica, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
    }
    h1 {
        font-size: 24pt;
        color: #2c3e50;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
    }
    h2 {
        font-size: 18pt;
        color: #34495e;
        margin-top: 20px;
    }
    h3 {
        font-size: 14pt;
        color: #7f8c8d;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        font-family: monospace;
    }
    pre {
        background-color: #f4f4f4;
        padding: 10px;
        border: 1px solid #ddd;
        white-space: pre-wrap;
    }
    img {
        max_width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    .mermaid {
        display: none; /* Mermaid diagrams often don't render in static PDF converters, hiding to avoid clutter */
    }
</style>
"""

def convert_report():
    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-process text specific fixups
    # Ensure images use simple relative paths if they aren't already
    # The current PROJECT_REPORT.md in root likely has simple paths "baseline.png"
    # xhtml2pdf resolves relative paths against the CWD, which is correct here.
    
    print("Converting Markdown to HTML...")
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    full_html = f"<html><head>{CSS}</head><body>{html_content}</body></html>"

    # Save HTML version
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved {OUTPUT_HTML}")

    # Convert to PDF
    print(f"Generating {OUTPUT_PDF}...")
    with open(OUTPUT_PDF, "wb") as f:
        pisa_status = pisa.CreatePDF(full_html, dest=f)

    if pisa_status.err:
        print("Error generating PDF")
    else:
        print(f"Successfully created {OUTPUT_PDF}")

if __name__ == "__main__":
    convert_report()
