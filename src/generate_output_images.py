#!/usr/bin/env python3
"""Generate 3 JPG summary images of the experiment results."""

from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

OUT_DIR = "/home/workdir/artifacts/prompt_eval_experiment/output"
os.makedirs(OUT_DIR, exist_ok=True)

# Try to load a monospace font; fall back to default
try:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    title_font = ImageFont.truetype(FONT_BOLD, 28)
    header_font = ImageFont.truetype(FONT_BOLD, 18)
    body_font = ImageFont.truetype(FONT_PATH, 15)
    small_font = ImageFont.truetype(FONT_PATH, 13)
except Exception:
    title_font = ImageFont.load_default()
    header_font = title_font
    body_font = title_font
    small_font = title_font


def create_image(width, height, bg=(248, 250, 252)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_rounded_rect(draw, xy, fill, radius=12):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


# ---------------------------------------------------------------------------
# Image 1 – Comparison Table
# ---------------------------------------------------------------------------
def make_table_image():
    W, H = 1100, 620
    img, draw = create_image(W, H)

    # Header bar
    draw.rectangle([0, 0, W, 70], fill=(30, 64, 175))
    draw.text((30, 18), "Prompt Evaluation Results  –  Climate Change RAG", font=title_font, fill="white")

    # Table data
    headers = ["Question", "Faith (A)", "Faith (B)", "Rel (A)", "Rel (B)"]
    rows = [
        ["What causes sea level rise?", "0.679", "0.775", "0.860", "0.940"],
        ["How do renewables help mitigate\nclimate change?", "0.720", "0.727", "0.767", "0.847"],
        ["What is the role of the IPCC?", "0.730", "0.941", "0.761", "0.706"],
        ["Why is the Arctic warming faster?", "0.793", "0.888", "0.831", "0.917"],
        ["AVERAGE", "0.731", "0.833", "0.805", "0.853"],
    ]

    col_x = [30, 480, 620, 760, 900]
    y = 100

    # Header row background
    draw.rounded_rectangle([20, y - 8, W - 20, y + 32], radius=6, fill=(226, 232, 240))
    for i, h in enumerate(headers):
        draw.text((col_x[i], y), h, font=header_font, fill=(30, 41, 59))
    y += 50

    for idx, row in enumerate(rows):
        bg = (241, 245, 249) if idx % 2 == 0 else (255, 255, 255)
        if idx == len(rows) - 1:
            bg = (219, 234, 254)  # highlight average
        draw.rounded_rectangle([20, y - 6, W - 20, y + 48], radius=6, fill=bg)
        for i, cell in enumerate(row):
            color = (15, 23, 42)
            if i > 0 and idx < len(rows) - 1:
                # colour B higher than A
                try:
                    a_val = float(rows[idx][1 if i in (1, 2) else 3])
                    b_val = float(rows[idx][2 if i in (1, 2) else 4])
                    if i in (2, 4) and b_val >= a_val:
                        color = (22, 163, 74)  # green for B
                except Exception:
                    pass
            draw.text((col_x[i], y + 6), cell, font=body_font, fill=color)
        y += 55

    # Footer note
    draw.text(
        (30, H - 40),
        "Faithfulness & Answer Relevancy (manual token-overlap approximation of RAGAS metrics)",
        font=small_font,
        fill=(100, 116, 139),
    )

    path = os.path.join(OUT_DIR, "01_comparison_table.jpg")
    img.save(path, "JPEG", quality=95)
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Image 2 – Analysis + Key Takeaways
# ---------------------------------------------------------------------------
def make_analysis_image():
    W, H = 1100, 680
    img, draw = create_image(W, H)

    draw.rectangle([0, 0, W, 70], fill=(5, 150, 105))
    draw.text((30, 18), "Analysis – Why Enhanced Prompt Wins", font=title_font, fill="white")

    y = 100
    paragraphs = [
        ("Key Finding", 
         "Prompt B (Enhanced) outperforms Prompt A on both faithfulness (+0.102 / +14 %) "
         "and answer relevancy (+0.048 / +6 %). Explicit instructions to stay grounded, "
         "remain concise and refuse speculation reduce unsupported statements."),
        ("Faithfulness Gain",
         "Baseline answers occasionally introduce plausible but context-absent claims "
         "(volcanic activity, magnetic fields, space weather, funding original research). "
         "These lower the faithfulness score. Prompt B stays strictly inside the retrieved "
         "context, producing higher overlap and fewer hallucinations."),
        ("Relevancy Gain",
         "Because Prompt B answers are shorter and more focused, a larger fraction of their "
         "tokens align with the ground-truth answer, improving the relevancy metric."),
        ("Practical Implication",
         "Adding a few carefully worded guard-rails to a RAG system prompt yields measurable "
         "quality improvements at almost zero extra inference cost."),
    ]

    for title, body in paragraphs:
        draw.rounded_rectangle([25, y, W - 25, y + 110], radius=10, fill=(236, 253, 245))
        draw.text((45, y + 12), title, font=header_font, fill=(6, 95, 70))
        # wrap body
        lines = textwrap.wrap(body, width=95)
        ty = y + 42
        for line in lines:
            draw.text((45, ty), line, font=body_font, fill=(30, 41, 59))
            ty += 20
        y += 125

    path = os.path.join(OUT_DIR, "02_analysis_summary.jpg")
    img.save(path, "JPEG", quality=95)
    print(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Image 3 – Ethical Risks & Mitigations
# ---------------------------------------------------------------------------
def make_ethics_image():
    W, H = 1100, 780
    img, draw = create_image(W, H)

    draw.rectangle([0, 0, W, 70], fill=(185, 28, 28))
    draw.text((30, 18), "Ethical & Security Risk Analysis", font=title_font, fill="white")

    risks = [
        ("1. Prompt Injection / Instruction Override",
         "Attacker embeds 'Ignore previous instructions…' inside the user question. "
         "Baseline prompt complies; Enhanced prompt refuses.",
         "Mitigation: High-priority system message + structured separators + optional "
         "injection classifier. Prompt B already contains an explicit non-override rule."),
        ("2. Hallucination / Unsupported Claims",
         "Model invents plausible climate facts absent from the retrieved context, "
         "potentially spreading misinformation.",
         "Mitigation: 'Do not speculate / say I don't know' instructions, continuous "
         "RAGAS faithfulness monitoring, multi-passage consistency checks, human review."),
        ("3. Bias Amplification & Incomplete Context",
         "Knowledge base under-represents certain regions or viewpoints → systematic "
         "under-serving of those perspectives.",
         "Mitigation: Diverse curated corpus, demographic audit logs, mandatory source "
         "citations for transparency."),
    ]

    y = 95
    colors = [(254, 226, 226), (254, 243, 199), (219, 234, 254)]
    title_colors = [(153, 27, 27), (146, 64, 14), (30, 64, 175)]

    for i, (title, desc, mit) in enumerate(risks):
        draw.rounded_rectangle([25, y, W - 25, y + 200], radius=12, fill=colors[i])
        draw.text((45, y + 15), title, font=header_font, fill=title_colors[i])
        # description
        for j, line in enumerate(textwrap.wrap(desc, width=95)):
            draw.text((45, y + 50 + j * 20), line, font=body_font, fill=(30, 41, 59))
        # mitigation
        draw.text((45, y + 110), "Mitigation:", font=header_font, fill=(15, 23, 42))
        for j, line in enumerate(textwrap.wrap(mit, width=95)):
            draw.text((45, y + 138 + j * 20), line, font=body_font, fill=(51, 65, 85))
        y += 220

    path = os.path.join(OUT_DIR, "03_ethical_risks.jpg")
    img.save(path, "JPEG", quality=95)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    make_table_image()
    make_analysis_image()
    make_ethics_image()
    print("All output images generated.")
