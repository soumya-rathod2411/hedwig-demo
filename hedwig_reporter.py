"""
hedwig_reporter.py

Runs once at 10pm (or on-demand, if triggered by the app). Reads
everything collected since the last report, writes a structured
intelligence report following the spec, saves it as PDF/txt, and
updates manifest.json -- the index the app reads to list every report
that's ever been generated.

Requires:
    pip install openai reportlab
"""

import json
import os
import re
import sys
from datetime import datetime
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Windows' default terminal encoding (cp1252) can't display many Unicode
# characters -- like the Rupee sign, curly quotes, em-dashes -- that a
# report can easily contain. Forcing UTF-8 here means print() never
# crashes on a character the terminal's default encoding doesn't support.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# =========================================================
# USE_CLOUD_MODEL is True automatically when running on GitHub Actions
# (which sets the GITHUB_ACTIONS env var) since there's no local GPU
# there -- no need to remember to flip this by hand for cloud runs.
# Locally, it stays False by default so you can keep testing with LM Studio.
USE_CLOUD_MODEL = os.environ.get("GITHUB_ACTIONS") == "true"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "nvapi-your_key_here").strip()
MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b" if USE_CLOUD_MODEL else "nvidia/nemotron-3-nano-4b"
# =========================================================

DATA_FOLDER = "data"
REPORTS_FOLDER = "reports"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

if USE_CLOUD_MODEL:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=300.0  # NVIDIA's hosted inference is fast -- no need for the long local timeout
    )
else:
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed-for-local",
        timeout=3600.0  # 1 hour -- generating locally on limited VRAM can be genuinely slow
    )


def sanitize_for_pdf(text):
    """
    reportlab's built-in fonts (Helvetica etc.) only support a limited
    character set, not full Unicode. Characters the AI likes to use --
    curly quotes, en/em dashes, and especially the Rupee sign -- have no
    glyph in these fonts and render as a blank black box instead. Swap
    them for safe equivalents every font can actually display.
    """
    replacements = {
        "\u2013": "-",    # en dash –
        "\u2014": "-",    # em dash —
        "\u2011": "-",    # non-breaking hyphen
        "\u2018": "'",    # left single quote '
        "\u2019": "'",    # right single quote '
        "\u201c": '"',    # left double quote "
        "\u201d": '"',    # right double quote "
        "\u2026": "...",  # ellipsis …
        "\u20b9": "Rs. ", # Rupee sign ₹
        "\u2022": "-",    # bullet • if the model writes one directly
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def build_pdf(report_text, pdf_path):
    """
    Converts the report's markdown-style text (## headers, **bold**,
    - bullets) into a formatted PDF. reportlab's Paragraph understands a
    small subset of HTML-like tags (<b>, etc.), so **bold** just needs
    converting to <b>bold</b> -- no full markdown library needed for
    something this simple.
    """
    report_text = sanitize_for_pdf(report_text)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("HedwigTitle", parent=styles["Title"], fontSize=18, spaceAfter=4)
    subtitle_style = ParagraphStyle("HedwigSubtitle", parent=styles["Normal"], fontSize=10,
                                     textColor=colors.grey, spaceAfter=20)
    h2_style = ParagraphStyle("HedwigH2", parent=styles["Heading2"], spaceBefore=16, spaceAfter=8,
                               textColor=colors.HexColor("#2c3e50"))
    h3_style = ParagraphStyle("HedwigH3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("HedwigBody", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=6)
    bullet_style = ParagraphStyle("HedwigBullet", parent=body_style, leftIndent=18)

    def inline_markdown(text):
        # Escape real HTML special characters first, THEN convert **bold**
        # to <b>bold</b> -- order matters, or escaping would mangle the tags.
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        return text

    elements = [
        Paragraph("Hedwig's 10PM Intelligence Report", title_style),
        Paragraph(datetime.now().strftime("%A, %d %B %Y"), subtitle_style),
    ]

    for raw_line in report_text.split("\n"):
        line = raw_line.strip()
        if not line:
            elements.append(Spacer(1, 6))
        elif line.startswith("### "):
            elements.append(Paragraph(inline_markdown(line[4:]), h3_style))
        elif line.startswith("## "):
            elements.append(Paragraph(inline_markdown(line[3:]), h2_style))
        elif line.startswith("---"):
            elements.append(Spacer(1, 10))
        elif line.startswith(("- ", "* ")):
            elements.append(Paragraph("&bull; " + inline_markdown(line[2:]), bullet_style))
        elif line.startswith("**") and line.endswith("**") and len(line) > 4:
            elements.append(Paragraph(inline_markdown(line), h3_style))
        else:
            elements.append(Paragraph(inline_markdown(line), body_style))

    doc.build(elements)

filename = os.path.join(DATA_FOLDER, "hedwig_data.jsonl")

try:
    with open(filename, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    entries = []

if not entries:
    print(f"No data collected since the last report ({filename} not found or empty). "
          f"Make sure hedwig_collector.py has been running.")
    sys.exit(1)  # explicit failure code -- so the scheduler knows nothing was sent

# --- Group entries by category so the model sees them pre-organized ---
by_category = {}
for e in entries:
    cat = e.get("category", "Uncategorized")
    by_category.setdefault(cat, []).append(e)

# Safety cap: truncate any single item's body/description text so one
# very long snippet can't blow up the total request size. This is a
# backstop independent of whatever context length is set in LM Studio --
# even if a future day collects much more data than usual, no single
# item can push the whole request over the limit by itself.
MAX_ITEM_CHARS = 400

def trim(text):
    text = text or ""
    return text if len(text) <= MAX_ITEM_CHARS else text[:MAX_ITEM_CHARS] + "..."

raw_text_parts = []
for category, items in by_category.items():
    raw_text_parts.append(f"\n=== {category} ===")
    for e in items:
        if e["type"] == "tavily":
            source = e.get("source", "web")
            published = f" ({e['published']})" if e.get("published") else ""
            raw_text_parts.append(f"[{source}{published}] {e['title']}: {trim(e['body'])}")
        elif e["type"] == "youtube":
            raw_text_parts.append(f"[YOUTUBE - {e['channel']}] {e['title']}: {trim(e['description'])}")
        elif e["type"] == "search_failed":
            raw_text_parts.append(f"[NOTE: search for this category failed and could not be retrieved today]")

raw_text = "\n".join(raw_text_parts)

print(f"Loaded {len(entries)} items across {len(by_category)} categories. Asking Hedwig to write tonight's report...\n")

# --- System prompt: implements the report structure and rules ---
system_prompt = """You are Hedwig, an intelligence filter for a Diploma IT student in India -- NOT a generic news summarizer.

CORE QUESTION for every item you include: Why should I care?
Prioritize impact, credibility, practical usefulness, relevance, and timeliness.
Prefer fewer high-value items over a large volume of news. Do not pad sections to fill a quota.
If nothing significant happened in a category, say so plainly instead of inventing filler.
If a category contains a note that its search failed today, mention briefly that today's data for that category was incomplete -- do not invent content to fill the gap.

Below is raw collected data, grouped by category, gathered periodically since the last report.
Many entries will be duplicates or near-duplicates -- consolidate them, don't list near-identical items separately.

Write tonight's report in EXACTLY this structure:

## 3 Things I Absolutely Should Know Today
The three highest-impact developments across ALL categories. Never invent items to fill this to three -- if there are only one or two truly important things, say so.

## AI
Models, agents, tools, research, APIs, companies, security, capability changes. Ignore trivial tool spam and repetitive benchmark news.

## Developer World
Programming, frameworks, APIs, GitHub, IDEs, cloud, databases, DevOps, deployment, testing, open source.

## Cybersecurity
Critical vulnerabilities, major breaches (only if they carry a broader lesson), attack trends, security tools. Focus on defensive understanding, not operational misuse detail.

## India
Policy, IndiaAI, MeitY, Digital India, privacy/data regulation, cybersecurity regulation, programs. For policies: explain what changed, who's affected, and whether the reader needs to act.

## Learn
Useful courses/resources, each ranked: Worth doing / Maybe / Skip. Prioritize skills and projects over certificate-collecting.

## Opportunities
Hackathons, competitions, internships, open source programs, fellowships, workshops, scholarships, free credits. For each: deadline, eligibility, online/location, cost, benefit, required skills, source link if available.

## Tools Worth Trying
Genuinely useful tools with a concrete practical use case -- not just "new tool exists."

## Tech Outside AI
Semiconductors, hardware, cloud, networking, major companies, other material developments.

## My Action for Tomorrow
Exactly ONE concrete action based on today's intelligence. Not a list -- one thing.

## If You Only Remember 3 Things
Three concise final takeaways.

FORMAT for each major item within a section:
**Headline**
What happened: 1-3 sentences.
Why you should care: specific to a Diploma IT student in India.
Impact: Low / Medium / High / Critical. -- NEVER omit this line, every single item must have it.
What to do: only include this line when genuinely justified.
Source: cite the source name from the data (e.g. "TechCrunch", "Reuters") -- do not fabricate a source not present in the data.

RULES:
- Write each section header (## AI, ## Cybersecurity, etc.) EXACTLY ONCE, in the order given above. NEVER repeat a header a second time, and NEVER write filler like "(covered above)" or "(no further news)" as if it were a new section.
- If a story genuinely belongs to two categories, mention it in full under the single most relevant category only, and simply omit it from the other category's section entirely -- do not reference it there at all.
- If a section genuinely has nothing significant, write one plain sentence saying so under that section's single header -- never a second header.
- Distinguish confirmed facts from company claims, opinions, or speculation when the data makes this clear.
- Do not treat a single unverified item as confirmed fact -- hedge appropriately if the data only gives one weak source.
- Do not recommend something solely because it appears frequently in the data.
- Do not encourage collecting certificates over building projects.
- Do not over-index on AI at the expense of core IT/developer fundamentals.
- Write in plain, direct language. No filler, no hype words."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Here is the raw collected data since the last report:\n{raw_text}\n\nWrite tonight's report."}
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=8000
)
report_text = response.choices[0].message.content

# --- Save the report: .txt as a plain backup, .pdf as the real deliverable ---
today = datetime.now().strftime("%Y-%m-%d")
report_filename = os.path.join(REPORTS_FOLDER, f"hedwig_report_{today}.txt")
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(f"Hedwig's 10PM Intelligence Report - {datetime.now().strftime('%A, %d %B %Y')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report_text)
print(f"Report saved to: {report_filename}")

pdf_filename = os.path.join(REPORTS_FOLDER, f"hedwig_report_{today}.pdf")
build_pdf(report_text, pdf_filename)
print(f"PDF built: {pdf_filename}")

# --- Update the manifest -- this is what the app reads to list every
# report that's ever been generated, for the date-range filtering. ---
MANIFEST_PATH = os.path.join(REPORTS_FOLDER, "manifest.json")

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_manifest(manifest):
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)

def extract_preview(text, max_chars=140):
    """Grabs the first real line of the report as a short preview
    the app can show in a list, without needing to open the full PDF."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped.lstrip("-*").strip())
            return cleaned[:max_chars]
    return "Hedwig Report"

try:
    manifest = load_manifest()
    manifest.append({
        "date": today,
        "pdf": os.path.basename(pdf_filename),
        "txt": os.path.basename(report_filename),
        "generated_at": datetime.now().isoformat(),
        "preview": extract_preview(report_text)
    })
    save_manifest(manifest)
    print(f"Manifest updated: {MANIFEST_PATH}")

    # Archive raw data (never delete permanently) and reset for next cycle.
    archive_filename = os.path.join(DATA_FOLDER, f"hedwig_data_archive_{today}.jsonl")
    os.replace(filename, archive_filename)
    with open(filename, "w", encoding="utf-8") as f:
        pass
    print(f"Raw data archived to: {archive_filename}")
    print(f"{filename} reset for the next collection cycle.")

except Exception as e:
    print(f"Failed to finalize report: {e}")
    print(f"(Report/PDF still saved to {report_filename} and {pdf_filename}. Raw data in {filename} "
          f"was NOT cleared -- nothing is lost.)")
    print("\n--- REPORT ---\n")
    print(report_text)
    sys.exit(1)

print("\n--- REPORT ---\n")
print(report_text)
sys.exit(0)
