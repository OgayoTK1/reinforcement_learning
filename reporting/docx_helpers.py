from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Calibri"
INK = RGBColor(0x22, 0x22, 0x22)


def new_document(body_pt: float = 9.5) -> Document:
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.8)
    s.top_margin, s.bottom_margin = Cm(1.5), Cm(1.5)
    st = d.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(body_pt); st.font.color.rgb = INK
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.paragraph_format.space_after = Pt(3); st.paragraph_format.space_before = Pt(0); st.paragraph_format.line_spacing = 1.04
    for name, size, before, after in (("Heading 1", 12.5, 8, 2), ("Heading 2", 10.5, 5, 1.5)):
        h = d.styles[name]
        h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = RGBColor(0x1b, 0x3a, 0x2a)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); h.element.rPr.rFonts.set(qn("w:ascii"), FONT); h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(after); h.paragraph_format.keep_with_next = True
    return d


def para(d, text: str = "", bold: bool = False, italic: bool = False, size: float | None = None, align=None, after: float | None = None, keep=False):
    p = d.add_paragraph()
    _runs(p, text, bold, italic, size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def _runs(p, text, bold=False, italic=False, size=None):
    """**bold** segments inside text."""
    text = text.replace("`", "")
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if not seg:
            continue
        r = p.add_run(seg); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size:
            r.font.size = Pt(size)


def shade(cell, hex_fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _borders(table, color="9a9a8a", sz=4):
    tbl = table._tbl
    tblPr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def table(d, header: list[str], rows: list[list[str]], widths_cm: list[float], size: float = 7.5, bold_rows=(), header_fill="E4DFC9", align_right_from: int | None = None):
    t = d.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _borders(t)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 15), ("bottom", 15), ("left", 50), ("right", 50)):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa"); mar.append(e)
    t._tbl.tblPr.append(mar)
    def fill(cell, text, bold=False, right=False):
        cell.text = ""
        p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
        if right:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _runs(p, str(text), bold=bold, size=size)
    for j, h in enumerate(header):
        fill(t.rows[0].cells[j], h, bold=True); shade(t.rows[0].cells[j], header_fill)
    trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for i, r in enumerate(rows):
        cells = t.add_row().cells
        for j, v in enumerate(r):
            fill(cells[j], v, bold=(i in bold_rows), right=(align_right_from is not None and j >= align_right_from))
        if i in bold_rows:
            for c in cells:
                shade(c, "EAF1E2")
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    for j, w in enumerate(widths_cm):
        t.columns[j].width = Cm(w)
    for row in t.rows:
        for j, w in enumerate(widths_cm):
            row.cells[j].width = Cm(w)
        cant = OxmlElement("w:cantSplit"); row._tr.get_or_add_trPr().append(cant)
    d.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def figure(d, path: Path, width_cm: float, caption: str):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.space_after = Pt(4)
    _runs(c, caption, italic=True, size=8)


def footer_page_numbers(d, left_text: str):
    sec = d.sections[0]
    p = sec.footer.paragraphs[0]; p.text = ""
    r = p.add_run(left_text + "    Page "); r.font.size = Pt(8)
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run(); run.font.size = Pt(8)
        if tag:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tag); run._r.append(f)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
