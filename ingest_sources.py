import os
import re
import json
import time
import hashlib
import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin
from collections import deque

import requests
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

TIMEOUT = 40
SLEEP_BETWEEN = 0.5  # be polite
BASE_DIR = Path(__file__).resolve().parent

VALID_JURISDICTIONS = {"england_wales", "scotland", "northern_ireland", "uk_wide"}

PDF_DOMAIN_ALLOWLIST = {
    "assets.publishing.service.gov.uk",
    "sentencingcouncil.org.uk",
    "www.justice-ni.gov.uk",
    "www.judiciaryni.uk",
    "assets.college.police.uk",
    "www.scottishsentencingcouncil.org.uk",
}

# ---- Minimal taggers (Tip 2) ----
def infer_source_org(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.endswith("sentencingcouncil.org.uk"):
        return "sentencing_council_england_wales"
    if host.endswith("college.police.uk") or host.endswith("assets.college.police.uk"):
        return "college_of_policing"
    if host.endswith("legislation.gov.uk"):
        return "legislation_gov_uk"
    if host.endswith("gov.uk") or host.endswith("assets.publishing.service.gov.uk"):
        return "gov_uk"
    if host.endswith("justice-ni.gov.uk"):
        return "doj_northern_ireland"
    if host.endswith("judiciaryni.uk"):
        return "judiciary_northern_ireland"
    if host.endswith("scottishsentencingcouncil.org.uk"):
        return "scottish_sentencing_council"
    if host.endswith("mygov.scot"):
        return "mygov_scot"
    return host or "other"

def infer_topic_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    last = path.split("/")[-1]
    last = re.sub(r"\.pdf$", "", last, flags=re.IGNORECASE)
    last = re.sub(r"[^a-zA-Z0-9]+", "_", last).strip("_").lower()
    return last or "unknown"

# ---- Minimal PDF preference (Tip 1) ----
def should_skip_html_when_pdfs_exist(url: str, source_org: str) -> bool:
    """
    Minimal heuristic: if PDFs exist on a page that is likely a guideline hub,
    skip embedding/saving the HTML to prefer PDFs.
    """
    u = url.lower()
    if source_org in {
        "sentencing_council_england_wales",
        "scottish_sentencing_council",
        "judiciary_northern_ireland",
        "doj_northern_ireland",
    }:
        return True
    # Also skip common "hub/index" pages where PDFs are the real content
    if "pace-codes-of-practice" in u or "pace-codes-practice" in u:
        return True
    return False


def url_host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def pick_primary_pdfs(html: str, base_url: str) -> list[str]:
    """
    Return a *ranked* list of PDF URLs likely to be the main document.
    (Kept from your current script; not strictly required for minimal version,
     but useful and already present.)
    """
    soup = BeautifulSoup(html, "lxml")
    candidates = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if not full.lower().endswith(".pdf"):
            continue

        host = url_host(full)
        if host and host not in PDF_DOMAIN_ALLOWLIST:
            continue

        text = (a.get_text(" ", strip=True) or "").lower()
        score = 0
        if "definitive" in text: score += 3
        if "guideline" in text: score += 3
        if "pdf" in text: score += 1
        if "final" in text or "web" in text: score += 1
        if "download" in text: score += 1
        if "sentencingcouncil.org.uk" in host: score += 2
        if "assets.publishing.service.gov.uk" in host: score += 2

        candidates.append((score, full))

    candidates.sort(key=lambda x: x[0], reverse=True)
    ranked = []
    seen = set()
    for score, u in candidates:
        if u not in seen:
            ranked.append(u)
            seen.add(u)
    return ranked


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:120] if len(s) > 120 else s

def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]

def is_pdf_url(url: str, content_type: str | None) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False

def extract_pdf_links_from_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if full.lower().endswith(".pdf"):
            pdfs.append(full)
    return list(dict.fromkeys(pdfs))

def looks_like_pdf_index_page(url: str, text: str, pdf_links: list[str]) -> bool:
    if not pdf_links:
        return False
    if "justice-ni.gov.uk/articles/pace-codes-practice" in url:
        return True
    if len(text) < 1200 and len(pdf_links) >= 2:
        return True
    return False

def extract_title_from_html(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
    except Exception:
        return None
    return None

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_url(url: str) -> tuple[bytes, str | None]:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type")
    return r.content, content_type

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    from io import BytesIO
    bio = BytesIO(pdf_bytes)
    reader = PdfReader(bio)
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append(t)
    return clean_text("\n\n".join(pages))

def extract_text_from_html(html_bytes: bytes, url: str) -> tuple[str, str | None]:
    html = html_bytes.decode("utf-8", errors="replace")

    extracted = trafilatura.extract(
        html,
        url=url,
        include_links=False,
        include_comments=False,
        include_tables=False,
        favor_recall=False,
    )

    title = extract_title_from_html(html)

    if extracted and extracted.strip():
        return clean_text(extracted), title

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return clean_text(text), title

def read_urls_file(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] in VALID_JURISDICTIONS and len(parts) >= 2:
                j = parts[0]
                url = parts[1]
            else:
                j = "uk_wide"
                url = parts[0]

            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError(f"URL missing scheme (http/https): {url} (line: {raw.rstrip()})")

            items.append((j, url))
    return items

def save_doc(out_dir: Path, doc: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{doc['id']}_{slugify(doc['url'])}.json"
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def relative_to_base(path: Path) -> str:
    return str(path.relative_to(BASE_DIR))

def canonicalize_url(url: str) -> str:
    u = url.strip()
    if "legislation.gov.uk" in u:
        if re.search(r"/section/\d+/?$", u):
            u = u.rstrip("/") + "/data.html"
        if "/article/" in u and "view=plain" not in u:
            sep = "&" if "?" in u else "?"
            u = u + f"{sep}view=plain"
    return u

def clean_legislation_text(text: str) -> str:
    m = re.search(r"(?m)^\(\s*1\s*\)", text)
    if m:
        text = text[m.start():]

    cut_markers = [
        "PrintThe Whole Act",
        "Print The Whole Act",
        "The Whole Act you have selected contains over",
        "Latest Available (revised):",
        "Geographical Extent:",
        "Show Timeline of Changes:"
    ]
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
            break
    return clean_text(text)

def split_legal_text_into_sections(
    text: str,
    min_chars: int = 800,
    max_chars: int = 3000
) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    blocks = []
    current = []

    number_pattern = re.compile(
        r"""^(
            (\(?\d+(\.\d+)*\)?)      # 1, 1.1, (1), (1.2)
            |
            (\d+\.)                  # 1.
        )\s+""",
        re.VERBOSE
    )
    caps_heading_pattern = re.compile(r"^[A-Z][A-Z \-]{6,}$")

    def flush():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        if number_pattern.match(line) or caps_heading_pattern.match(line):
            flush()
        current.append(line)
    flush()

    merged = []
    buf = ""
    for block in blocks:
        if not buf:
            buf = block
        elif len(buf) < min_chars:
            buf = buf + "\n\n" + block
        else:
            merged.append(buf)
            buf = block
    if buf:
        merged.append(buf)

    final = []
    for m in merged:
        if len(m) <= max_chars:
            final.append(m)
        else:
            start = 0
            overlap = 300
            while start < len(m):
                end = min(len(m), start + max_chars)
                final.append(m[start:end])
                if end == len(m):
                    break
                start = end - overlap
    return final


def main():
    urls_path = BASE_DIR / "urls.txt"
    out_base = BASE_DIR / "sources"
    manifest = []

    seed_items = read_urls_file(urls_path)
    print(f"Loaded {len(seed_items)} URL(s)")

    q = deque((j, u) for j, u in seed_items)
    seen_urls: set[str] = set()

    while q:
        jurisdiction, url = q.popleft()
        url = canonicalize_url(url)

        if url in seen_urls:
            continue
        seen_urls.add(url)

        print(f"\nFetching [{jurisdiction}] {url}")
        try:
            content, content_type = fetch_url(url)
            retrieved_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

            pdf = is_pdf_url(url, content_type)
            out_dir = out_base / jurisdiction

            # Minimal metadata (Tip 2)
            source_org = infer_source_org(url)
            topic = infer_topic_from_url(url)

            if pdf:
                raw_text = extract_text_from_pdf(content)
                if "legislation.gov.uk" in url and raw_text:
                    raw_text = clean_legislation_text(raw_text)

                sections = split_legal_text_into_sections(raw_text)
                if not sections:
                    sections = [raw_text]

                for i, section in enumerate(sections):
                    doc = {
                        "id": stable_id(f"{url}#{i}"),
                        "title": f"{urlparse(url).path.split('/')[-1] or 'PDF'} — section {i+1}",
                        "url": url,
                        "jurisdiction": jurisdiction,
                        "retrieved_at": retrieved_at,
                        "content_type": content_type,
                        "text": section or "",
                        "source_org": source_org,
                        "topic": topic,
                        "chunk_index": i,
                    }
                    path = save_doc(out_dir, doc)
                    manifest.append({"jurisdiction": jurisdiction, "url": url, "file": relative_to_base(path)})
                print(f"  ✅ Saved {len(sections)} PDF section(s)")

            else:
                html = content.decode("utf-8", errors="replace")

                # 1) discover PDFs and enqueue them
                pdf_links = extract_pdf_links_from_html(html, url)
                for pdf_url in pdf_links:
                    pdf_url = canonicalize_url(pdf_url)
                    if pdf_url not in seen_urls:
                        q.append((jurisdiction, pdf_url))

                # 2) extract text from the HTML
                text, title = extract_text_from_html(content, url)

                if "legislation.gov.uk" in url and text:
                    text = clean_legislation_text(text)

                if not text:
                    text = ""

                # Decide whether to save HTML
                skip_save = False

                # Existing index-page heuristic
                if looks_like_pdf_index_page(url, text, pdf_links):
                    skip_save = True
                    print("  ℹ️ Looks like a PDF index page; skipping save (PDFs will be ingested).")

                # Minimal PDF preference heuristic (Tip 1)
                if (not skip_save) and pdf_links and should_skip_html_when_pdfs_exist(url, source_org):
                    skip_save = True
                    print("  ℹ️ PDFs found on a guideline/hub page; skipping HTML save to prefer PDFs.")

                if not skip_save:
                    if len(text) < 200:
                        print("  ⚠️ Extracted text looks short; you may want to inspect this source.")

                    doc = {
                        "id": stable_id(url),
                        "title": title or urlparse(url).netloc,
                        "url": url,
                        "jurisdiction": jurisdiction,
                        "retrieved_at": retrieved_at,
                        "content_type": content_type,
                        "text": text,
                        "source_org": source_org,
                        "topic": topic,
                    }
                    path = save_doc(out_dir, doc)
                    manifest.append({"jurisdiction": jurisdiction, "url": url, "file": relative_to_base(path)})
                    print(f"  ✅ Saved: {path}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

        time.sleep(SLEEP_BETWEEN)

    out_base.mkdir(exist_ok=True)
    manifest_path = out_base / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
