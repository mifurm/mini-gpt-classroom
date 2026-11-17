"""Render a shared lecture specification as editable PowerPoint and HTML."""

from html import escape
import math

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt


COLORS = ["#087f5b", "#2274a5", "#a652b1", "#d18400", "#d34f47", "#586778", "#236a73"]
INK = "#172b42"
PAPER = "#fffdf8"
MUTED = "#617084"


def rgb(value):
    return RGBColor.from_string(value.lstrip("#"))


def chart_svg(block):
    width, height = 1080, 330
    left, right, top, bottom = 75, 25, 18, 50
    series = block["series"]
    values = [value for item in series for _, value in item["points"]]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.12, 0.03)
    low, high = low - margin, high + margin
    max_x = max(x for item in series for x, _ in item["points"])
    min_x = min(x for item in series for x, _ in item["points"])

    def xy(x, y):
        return (left + (x - min_x) / max(1, max_x - min_x) * (width - left - right),
                top + (high - y) / (high - low) * (height - top - bottom))

    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(block["label"])}">'
        '<rect width="100%" height="100%" fill="white"/>'
    ]
    for step in range(5):
        value = low + (high - low) * step / 4
        _, y = xy(min_x, value)
        content.append(f'<path d="M{left} {y:.1f} H{width-right}" stroke="#d9e0e5"/>'
                       f'<text x="4" y="{y+6:.1f}" font-size="17">{value:.2f}</text>')
    for step in range(5):
        value = min_x + (max_x - min_x) * step / 4
        x, _ = xy(value, low)
        content.append(f'<text x="{x-16:.1f}" y="{height-18}" font-size="17">{value:.0f}</text>')
    for index, item in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                          (xy(x, y) for x, y in item["points"]))
        content.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in item["points"]:
            x, y = xy(x, y)
            content.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    content.append("</svg>")
    legend = " · ".join(
        f'<span style="color:{COLORS[index % len(COLORS)]}">{escape(item["name"])}</span>'
        for index, item in enumerate(series))
    return "".join(content) + f'<div class="legend">{legend} · {escape(block["label"])}</div>'


def diagram_svg(block):
    labels = block["labels"]
    count = len(labels)
    width, height = 1080, 210
    gap = 30
    box_width = (width - 20 - gap * (count - 1)) / count
    marker = "arrow-" + "".join(character for character in block["label"] if character.isalnum())
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{escape(block["label"])}">'
                f'<defs><marker id="{marker}" markerWidth="8" markerHeight="8" '
                'refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" '
                'fill="#087f5b"/></marker></defs>']
    for index, label in enumerate(labels):
        x = 10 + index * (box_width + gap)
        elements.append(f'<rect x="{x:.1f}" y="20" width="{box_width:.1f}" height="148" '
                        'rx="9" fill="#e8f4ee" stroke="#087f5b" stroke-width="2"/>')
        lines = label.split("\n")
        for line_index, line in enumerate(lines):
            y = 76 + (line_index - (len(lines) - 1) / 2) * 25
            elements.append(f'<text x="{x+box_width/2:.1f}" y="{y:.1f}" text-anchor="middle" '
                            f'font-size="20" fill="{INK}">{escape(line)}</text>')
        if index < count - 1:
            elements.append(f'<path d="M{x+box_width+3:.1f} 92 h{gap-10}" '
                            f'stroke="#087f5b" stroke-width="2" marker-end="url(#{marker})"/>')
    elements.append("</svg>")
    return "".join(elements)


def html_block(block):
    kind = block["kind"]
    if kind == "text":
        return f'<p>{escape(block["text"])}</p>'
    if kind == "points":
        return "<ul>" + "".join(f"<li>{escape(text)}</li>" for text in block["items"]) + "</ul>"
    if kind == "code":
        return (f'<div class="source">{escape(block["source"])}</div>'
                f'<pre><code>{escape(block["text"])}</code></pre>')
    if kind == "table":
        header = "".join(f"<th>{escape(str(value))}</th>" for value in block["headers"])
        rows = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) +
                       "</tr>" for row in block["rows"])
        return f'<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'
    if kind == "chart":
        return chart_svg(block)
    if kind == "diagram":
        return diagram_svg(block)
    if kind == "mask":
        size = block["size"]
        header = "<th>Q / K</th>" + "".join(f"<th>k{i}</th>" for i in range(size))
        rows = "".join(
            f"<tr><td>q{row}</td>" + "".join(
                f'<td class="{"allowed" if column <= row else "blocked"}">{int(column <= row)}</td>'
                for column in range(size)) + "</tr>" for row in range(size))
        return f'<div class="causal"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
    if kind == "sample":
        return (f'<div class="sample-label">{escape(block["label"])}</div>'
                f'<blockquote>{escape(block["text"])}</blockquote>')
    if kind == "sample_grid":
        return (f'<div class="sample-grid" style="grid-template-columns:repeat({len(block["items"])},1fr)">'
                + "".join('<div class="sample-cell">' + html_block(dict(kind="sample", **item)) +
                          "</div>" for item in block["items"]) +
                "</div>")
    raise ValueError(f"Unknown slide block: {kind}")


def render_html(slides, title):
    sections = []
    for index, slide in enumerate(slides):
        blocks = "".join(html_block(block) for block in slide["blocks"])
        sections.append(
            f'<section class="slide{" active" if index == 0 else ""}" id="slide-{index+1}">'
            f'<div class="eyebrow">{escape(slide["section"])}</div>'
            f'<h1>{escape(slide["title"])}</h1><div class="content">{blocks}</div>'
            f'<aside class="notes"><strong>Speaker notes</strong><p>{escape(slide["notes"])}</p>'
            f'</aside><footer>{index+1} / {len(slides)}</footer></section>'
        )
    options = "".join(f'<option value="{index}">{index+1}. {escape(slide["title"])}</option>'
                      for index, slide in enumerate(slides))
    css = """
*{box-sizing:border-box}body{margin:0;background:#172b42;color:#172b42;
font-family:Arial,Helvetica,sans-serif}.slide{display:none;position:relative;width:1280px;
height:720px;margin:24px auto;background:#fffdf8;padding:34px 56px 48px;
border-top:7px solid #087f5b}.slide.active{display:block}.eyebrow{font-size:13px;
color:#087f5b;text-transform:uppercase;letter-spacing:1.7px;font-weight:700}
h1{font-size:34px;line-height:1.14;margin:13px 0 24px;max-height:78px}
.content{font-size:22px;line-height:1.35}p{margin:12px 0}ul{padding-left:30px;
margin:14px 0}li{margin:10px 0}.source{font-size:13px;color:#617084;
font-family:monospace;margin-top:12px}pre{font-size:18px;line-height:1.36;
background:#eef2f5;border-left:4px solid #087f5b;padding:15px;margin:6px 0 14px;
white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,Menlo,monospace}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:17px;
line-height:1.3;table-layout:auto}td,th{padding:10px 8px;border-bottom:1px solid #d5dedc;
text-align:left;vertical-align:middle}th{background:#e8f4ee;font-weight:700}
tbody tr:nth-child(even){background:#f1f4f5}svg{display:block;width:100%;height:auto;
margin:12px 0}.legend{font-size:15px;line-height:1.4;margin:4px 0 14px}
.sample-label{font-size:15px;font-weight:700;color:#087f5b;margin:15px 0 4px}
blockquote{background:#eef2f5;white-space:pre-wrap;font-size:19px;line-height:1.35;
padding:12px 16px;border-left:4px solid #087f5b;margin:0 0 16px}
.sample-grid{display:grid;gap:18px}.sample-grid blockquote{font-size:18px}
.causal{max-width:670px;margin:auto}.causal td:not(:first-child){text-align:center;
font-family:monospace}.causal td.allowed{background:#c6ebd8}
.causal td.blocked{background:#e0e6eb;color:#617084}.causal th{font-family:monospace}
.notes{display:none;background:#f6efda;padding:18px;margin-top:25px;font-size:17px;
line-height:1.5}body.show-notes .slide{height:auto;min-height:720px}
body.show-notes .notes{display:block}footer{position:absolute;bottom:20px;right:36px;
font-size:13px;color:#617084}nav{display:flex;justify-content:center;align-items:center;
gap:12px;padding:0 15px 22px;color:white}button,select{padding:10px;border:0;
border-radius:4px;font-size:14px}select{max-width:500px}button{cursor:pointer}
@media(max-width:1280px){.slide{width:100%;height:auto;min-height:720px;margin:0}}
@media print{@page{size:13.333333in 7.5in;margin:0}body{background:white}
.slide{display:block!important;width:1280px;height:720px;min-height:0;margin:0;
break-after:page;break-inside:avoid}.notes,nav{display:none!important}}
"""
    js = """
const slides=[...document.querySelectorAll('.slide')];let index=0;
function show(i){index=Math.max(0,Math.min(slides.length-1,i));
slides.forEach((s,n)=>s.classList.toggle('active',n===index));
document.querySelector('#counter').textContent=`${index+1} / ${slides.length}`;
document.querySelector('#jump').value=index;history.replaceState(null,'',`#slide-${index+1}`);}
document.querySelector('#prev').onclick=()=>show(index-1);
document.querySelector('#next').onclick=()=>show(index+1);
document.querySelector('#jump').onchange=e=>show(Number(e.target.value));
document.querySelector('#notes').onclick=()=>document.body.classList.toggle('show-notes');
document.addEventListener('keydown',e=>{
if(e.target.tagName==='SELECT')return;
if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1);}
if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1);}
if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);
if(e.key.toLowerCase()==='n')document.body.classList.toggle('show-notes');});
const match=location.hash.match(/^#slide-(\\d+)$/);show(match?Number(match[1])-1:0);
"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{css}</style></head><body>' +
            "".join(sections) + '<nav aria-label="Lecture navigation"><button id="prev">Previous</button>'
            '<span id="counter"></span><button id="next">Next</button><select id="jump" '
            f'aria-label="Jump to slide">{options}</select><button id="notes">Notes (N)</button>'
            '</nav><script>' + js + '</script></body></html>\n')


def text_box(slide, x, y, width, height, text, size=20, color=INK, bold=False, mono=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.03)
    frame.margin_top = frame.margin_bottom = Inches(0.03)
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Courier New" if mono else "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = rgb(color)
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.12
    return shape


def block_height(block):
    kind = block["kind"]
    if kind == "text":
        lines = sum(max(1, math.ceil(len(line) / 92)) for line in block["text"].split("\n"))
        return 0.32 * lines + 0.15
    if kind == "points":
        lines = sum(max(1, math.ceil(len(line) / 92)) for line in block["items"])
        return 0.33 * lines + 0.13 * len(block["items"])
    if kind == "code":
        lines = sum(max(1, math.ceil(len(line) / 90)) for line in block["text"].splitlines())
        return 0.245 * lines + 0.49
    if kind == "table":
        _, heights = table_layout(block)
        return sum(heights) + 0.15
    if kind == "chart":
        return 3.95
    if kind == "diagram":
        return 2.25
    if kind == "mask":
        return 0.46 * (block["size"] + 1) + 0.15
    if kind == "sample":
        lines = sum(max(1, math.ceil(len(line) / 100)) for line in block["text"].splitlines())
        return 0.275 * lines + 0.48
    if kind == "sample_grid":
        limit = 104 // len(block["items"])
        lines = max(sum(max(1, math.ceil(len(line) / limit)) for line in item["text"].splitlines())
                    for item in block["items"])
        return 0.26 * lines + 0.62
    raise ValueError(kind)


def table_layout(block):
    rows = [block["headers"]] + block["rows"]
    size = 14.5 if len(block["headers"]) >= 6 else 16
    weights = [min(42, max(6, max(len(str(row[column])) for row in rows)))
               for column in range(len(block["headers"]))]
    widths = [12.17 * weight / sum(weights) for weight in weights]
    heights = []
    for row in rows:
        lines = max(max(1, math.ceil(len(str(value)) / (width * 72 / (size * 0.55))))
                    for value, width in zip(row, widths))
        heights.append(max(0.42, lines * size * 1.15 / 72 + 0.12))
    return widths, heights


def pptx_block(slide, block, y, height):
    x, width = 0.58, 12.17
    kind = block["kind"]
    if kind in {"text", "points"}:
        text = block["text"] if kind == "text" else "\n".join("• " + item for item in block["items"])
        text_box(slide, x, y, width, height, text, size=19)
    elif kind == "code":
        text_box(slide, x, y, width, 0.24, block["source"], size=10.5, color=MUTED)
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y+0.28),
                                       Inches(width), Inches(height-0.34))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb("#eef2f5")
        shape.line.fill.background()
        text_box(slide, x+0.12, y+0.3, width-0.24, height-0.35, block["text"], size=15.5, mono=True)
    elif kind in {"table", "mask"}:
        if kind == "mask":
            size = block["size"]
            headers = ["Q / K"] + [f"k{i}" for i in range(size)]
            rows = [[f"q{i}"] + [str(int(j <= i)) for j in range(size)] for i in range(size)]
            x, width = 3.0, 7.2
        else:
            headers, rows = block["headers"], block["rows"]
        table = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y),
                                      Inches(width), Inches(height-0.12)).table
        if kind == "table":
            widths, heights = table_layout(block)
            for column, value in zip(table.columns, widths):
                column.width = Inches(value)
            for row, value in zip(table.rows, heights):
                row.height = Inches(value)
        for i, row in enumerate([headers] + rows):
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.text = str(value)
                cell.margin_left = cell.margin_right = Inches(0.07)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                fill = "#e8f4ee" if i == 0 else ("#f1f4f5" if i % 2 == 0 else PAPER)
                if kind == "mask" and i > 0 and j > 0:
                    fill = "#c6ebd8" if j <= i else "#e0e6eb"
                cell.fill.fore_color.rgb = rgb(fill)
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.name = "Arial"
                    paragraph.font.size = Pt(14.5 if len(headers) >= 6 else 16)
                    paragraph.font.bold = i == 0
                    paragraph.font.color.rgb = rgb(INK)
    elif kind == "diagram":
        labels = block["labels"]
        gap = 0.32
        box_width = (width - gap * (len(labels)-1)) / len(labels)
        for index, label in enumerate(labels):
            left = x + index * (box_width + gap)
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                           Inches(y+0.15), Inches(box_width), Inches(1.58))
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb("#e8f4ee")
            shape.line.color.rgb = rgb(COLORS[0])
            text_box(slide, left+0.06, y+0.5, box_width-0.12, 1.05, label, size=16, bold=True)
            if index < len(labels)-1:
                arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left+box_width+0.04),
                                               Inches(y+0.78), Inches(gap-0.08), Inches(0.28))
                arrow.fill.solid()
                arrow.fill.fore_color.rgb = rgb(COLORS[0])
                arrow.line.fill.background()
    elif kind == "chart":
        categories = sorted({point[0] for item in block["series"] for point in item["points"]})
        data = CategoryChartData()
        data.categories = [str(value) for value in categories]
        for item in block["series"]:
            mapping = dict(item["points"])
            data.add_series(item["name"], [mapping.get(value) for value in categories])
        chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y),
                                      Inches(width), Inches(height-0.34), data).chart
        chart.has_legend = True
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
        chart.value_axis.tick_labels.font.size = Pt(12)
        chart.category_axis.tick_labels.font.size = Pt(12)
        chart.value_axis.tick_labels.number_format = "0.00"
        for index, series in enumerate(chart.series):
            series.format.line.color.rgb = rgb(COLORS[index % len(COLORS)])
            series.format.line.width = Pt(2.2)
        text_box(slide, x, y+height-0.26, width, 0.26, block["label"], size=11, color=MUTED)
    elif kind == "sample":
        text_box(slide, x, y, width, 0.25, block["label"], size=12, color=COLORS[0], bold=True)
        text_box(slide, x+0.1, y+0.31, width-0.2, height-0.33, block["text"], size=17)
    elif kind == "sample_grid":
        count = len(block["items"])
        gap = 0.18
        column = (width - (count-1)*gap) / count
        for index, item in enumerate(block["items"]):
            left = x + index * (column+gap)
            text_box(slide, left, y, column, 0.46, item["label"],
                     size=11, color=COLORS[0], bold=True)
            text_box(slide, left, y+0.5, column, height-0.5, item["text"], size=15.5)
    else:
        raise ValueError(kind)


def render_pptx(slides, path):
    presentation = Presentation()
    presentation.slide_width = Inches(13.333333)
    presentation.slide_height = Inches(7.5)
    for index, item in enumerate(slides):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(PAPER)
        text_box(slide, 0.6, 0.24, 12.1, 0.3, item["section"].upper(),
                 size=10.5, color=COLORS[0], bold=True)
        text_box(slide, 0.58, 0.64, 12.17, 0.57, item["title"], size=29, bold=True)
        y = 1.42
        for block in item["blocks"]:
            height = block_height(block)
            if y + height > 7.05:
                raise ValueError(f"PowerPoint overflow on slide {index+1}: {item['title']} "
                                 f"({y+height:.2f} inches)")
            pptx_block(slide, block, y, height)
            y += height + 0.06
        text_box(slide, 11.85, 7.14, 0.88, 0.22, f"{index+1} / {len(slides)}",
                 size=10, color=MUTED)
        slide.notes_slide.notes_text_frame.text = (
            f"{index+1}. {item['title']}\n\n{item['notes']}"
        )
    presentation.save(path)
