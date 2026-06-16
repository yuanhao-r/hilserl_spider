#!/usr/bin/env python3
"""Browse replay-buffer transition pickle files in a local web page."""

from __future__ import annotations

import argparse
import html
import pickle
import struct
import time
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png_rgb(image: np.ndarray) -> bytes:
    image = np.asarray(image)
    while image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f"unsupported image shape {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = np.ascontiguousarray(image)
    color_type = 6 if image.shape[2] == 4 else 2
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=3))
        + _png_chunk(b"IEND", b"")
    )


def is_image_array(value: Any) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim in (3, 4)
        and value.shape[-1] in (1, 3, 4)
    )


def iter_images(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_images(child, prefix + (str(key),))
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            yield "/".join(prefix), value
        elif value.ndim == 4 and value.shape[-1] in (1, 3, 4):
            for index, image in enumerate(value):
                yield "/".join(prefix + (f"{index:02d}",)), image


def summarize_value(value: Any, depth: int = 0) -> str:
    if depth > 4:
        return html.escape(repr(type(value)))
    if is_image_array(value):
        arr = np.asarray(value)
        return f"image array shape={arr.shape} dtype={arr.dtype}"
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = []
        for key, child in value.items():
            rows.append(
                "<details open>"
                f"<summary>{html.escape(str(key))}</summary>"
                f"{summarize_value(child, depth + 1)}"
                "</details>"
            )
        return "\n".join(rows)
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            return f"{type(value).__name__}(len={len(value)})"
        return html.escape(repr(value))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        flat = arr.reshape(-1)
        if flat.size <= 32:
            body = np.array2string(arr, precision=4, suppress_small=True)
        else:
            head = np.array2string(flat[:16], precision=4, suppress_small=True)
            body = f"{head} ... ({flat.size} values)"
        return (
            f"<code>ndarray shape={arr.shape} dtype={arr.dtype}</code>"
            f"<pre>{html.escape(body)}</pre>"
        )
    if isinstance(value, (np.generic,)):
        return html.escape(repr(value.item()))
    return html.escape(repr(value))


class PickleCache:
    def __init__(self, max_items: int = 3):
        self.max_items = int(max_items)
        self._items: dict[Path, tuple[float, float, Any]] = {}

    def load(self, path: Path) -> Any:
        mtime = path.stat().st_mtime
        cached = self._items.get(path)
        if cached is not None and cached[0] == mtime:
            self._items[path] = (mtime, time.time(), cached[2])
            return cached[2]
        with path.open("rb") as f:
            data = pickle.load(f)
        self._items[path] = (mtime, time.time(), data)
        if len(self._items) > self.max_items:
            oldest = min(self._items, key=lambda p: self._items[p][1])
            self._items.pop(oldest, None)
        return data


class ReplayBufferBrowser:
    def __init__(self, buffer_dir: Path, cache_size: int = 3):
        self.buffer_dir = buffer_dir.resolve()
        self.cache = PickleCache(max_items=cache_size)

    def pkl_files(self) -> list[Path]:
        if self.buffer_dir.is_file():
            return [self.buffer_dir]
        return sorted(self.buffer_dir.glob("*.pkl"))

    def file_for_id(self, file_id: str | None) -> Path | None:
        files = self.pkl_files()
        if not files:
            return None
        if file_id is None:
            return files[0]
        try:
            idx = int(file_id)
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        for path in files:
            if path.name == file_id:
                return path
        return files[0]

    def file_index(self, path: Path) -> int:
        files = self.pkl_files()
        for idx, item in enumerate(files):
            if item == path:
                return idx
        return 0

    def transitions(self, path: Path) -> list:
        data = self.cache.load(path)
        if isinstance(data, list):
            return data
        if isinstance(data, tuple):
            return list(data)
        raise TypeError(f"{path.name} does not contain a transition list, got {type(data)}")


def parse_query(path: str) -> tuple[str, dict[str, list[str]]]:
    parsed = urllib.parse.urlparse(path)
    return parsed.path, urllib.parse.parse_qs(parsed.query)


def q_one(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    return values[0] if values else default


def make_handler(browser: ReplayBufferBrowser):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

        def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, body: str, status: int = 200) -> None:
            self.send_bytes(status, "text/html; charset=utf-8", body.encode("utf-8"))

        def send_error_html(self, status: int, message: str) -> None:
            self.send_html(
                f"<h1>{status}</h1><pre>{html.escape(message)}</pre>",
                status=status,
            )

        def do_GET(self):
            route, query = parse_query(self.path)
            try:
                if route == "/image":
                    self.handle_image(query)
                elif route == "/fragment":
                    self.handle_fragment(query)
                elif route == "/":
                    self.handle_index(query)
                else:
                    self.send_error_html(404, "not found")
            except Exception as exc:
                self.send_error_html(500, str(exc))

        def handle_image(self, query: dict[str, list[str]]) -> None:
            pkl_path = browser.file_for_id(q_one(query, "file"))
            if pkl_path is None:
                raise FileNotFoundError(f"no .pkl files in {browser.buffer_dir}")
            transitions = browser.transitions(pkl_path)
            idx = int(q_one(query, "idx", "0") or "0")
            image_path = q_one(query, "path", "") or ""
            if not (0 <= idx < len(transitions)):
                raise IndexError(f"transition index out of range: {idx}")
            for path, image in iter_images(transitions[idx]):
                if path == image_path:
                    self.send_bytes(200, "image/png", encode_png_rgb(image))
                    return
            raise KeyError(f"image not found: {image_path}")

        def handle_index(self, query: dict[str, list[str]]) -> None:
            files = browser.pkl_files()
            if not files:
                self.send_html(render_empty(browser.buffer_dir))
                return
            pkl_path = browser.file_for_id(q_one(query, "file"))
            assert pkl_path is not None
            file_idx = browser.file_index(pkl_path)
            transitions = browser.transitions(pkl_path)
            idx = max(0, min(int(q_one(query, "idx", "0") or "0"), len(transitions) - 1))
            body = render_page(browser, files, file_idx, pkl_path, transitions, idx)
            self.send_html(body)

        def handle_fragment(self, query: dict[str, list[str]]) -> None:
            files = browser.pkl_files()
            if not files:
                self.send_html(render_empty(browser.buffer_dir))
                return
            pkl_path = browser.file_for_id(q_one(query, "file"))
            assert pkl_path is not None
            file_idx = browser.file_index(pkl_path)
            transitions = browser.transitions(pkl_path)
            idx = max(0, min(int(q_one(query, "idx", "0") or "0"), len(transitions) - 1))
            self.send_html(render_page_body(browser, files, file_idx, pkl_path, transitions, idx))

    return Handler


def page_link(file_idx: int, idx: int, text: str) -> str:
    href = f"/?file={file_idx}&idx={idx}"
    return f'<a href="{html.escape(href)}">{html.escape(text)}</a>'


def render_empty(buffer_dir: Path) -> str:
    return html_doc(
        "Replay Buffer Browser",
        f"""
        <main>
          <h1>No pickle files found</h1>
          <p>Directory: <code>{html.escape(str(buffer_dir))}</code></p>
        </main>
        """,
    )


def render_page(
    browser: ReplayBufferBrowser,
    files: list[Path],
    file_idx: int,
    pkl_path: Path,
    transitions: list,
    idx: int,
) -> str:
    return html_doc(
        "Replay Buffer Browser",
        render_page_body(browser, files, file_idx, pkl_path, transitions, idx),
    )


def render_page_body(
    browser: ReplayBufferBrowser,
    files: list[Path],
    file_idx: int,
    pkl_path: Path,
    transitions: list,
    idx: int,
) -> str:
    transition = transitions[idx]
    image_cards = []
    for image_path, image in iter_images(transition):
        src = (
            f"/image?file={file_idx}&idx={idx}&path="
            f"{urllib.parse.quote(image_path, safe='')}"
        )
        arr = np.asarray(image)
        display_arr = arr
        while display_arr.ndim == 4 and display_arr.shape[0] == 1:
            display_arr = display_arr[0]
        height, width = display_arr.shape[:2] if display_arr.ndim >= 2 else (1, 1)
        image_cards.append(
            f"""
            <figure>
              <div class="image-frame" style="aspect-ratio: {int(width)} / {int(height)};">
                <img src="{html.escape(src)}" loading="lazy" width="{int(width)}" height="{int(height)}">
              </div>
              <figcaption>{html.escape(image_path)}<br>
                <code>{arr.shape} {arr.dtype}</code>
              </figcaption>
            </figure>
            """
        )

    options = "\n".join(
        f'<option value="{i}" {"selected" if i == file_idx else ""}>{html.escape(path.name)}</option>'
        for i, path in enumerate(files)
    )
    prev_idx = max(0, idx - 1)
    next_idx = min(len(transitions) - 1, idx + 1)
    jump_values = [0, max(0, idx - 100), prev_idx, idx, next_idx, min(len(transitions) - 1, idx + 100), len(transitions) - 1]
    jump_links = " ".join(
        page_link(file_idx, value, str(value)) for value in dict.fromkeys(jump_values)
    )

    top_level = ""
    if isinstance(transition, dict):
        rows = []
        for key, value in transition.items():
            if key in ("observations", "next_observations"):
                continue
            rows.append(
                f"<section><h3>{html.escape(str(key))}</h3>{summarize_value(value)}</section>"
            )
        top_level = "\n".join(rows)

    obs_summary = ""
    if isinstance(transition, dict):
        for key in ("observations", "next_observations", "infos", "info"):
            if key in transition:
                obs_summary += (
                    f"<section><h2>{html.escape(key)}</h2>"
                    f"{summarize_value(transition[key])}</section>"
                )
    if not obs_summary:
        obs_summary = summarize_value(transition)

    content = f"""
    <aside>
      <h1>Replay Buffer</h1>
      <form action="/" method="get">
        <label>File</label>
        <select name="file" onchange="this.form.submit()">{options}</select>
      </form>
      <p><code>{html.escape(str(pkl_path))}</code></p>
      <p>{len(transitions)} transitions</p>
      <nav>
        {page_link(file_idx, 0, "First")}
        {page_link(file_idx, prev_idx, "Prev")}
        {page_link(file_idx, next_idx, "Next")}
        {page_link(file_idx, len(transitions) - 1, "Last")}
      </nav>
      <p class="jumps">Jump: {jump_links}</p>
    </aside>
    <main>
      <header>
        <h1>{html.escape(pkl_path.name)} / transition {idx}</h1>
      </header>
      <section>
        <h2>Images</h2>
        <div class="image-grid">{''.join(image_cards) or '<p>No image arrays found.</p>'}</div>
      </section>
      <section>
        <h2>Transition</h2>
        {top_level}
      </section>
      {obs_summary}
    </main>
    <form class="timeline" action="/" method="get">
      <input type="hidden" name="file" value="{file_idx}">
      <button type="button" class="timeline-nav" data-idx="0">First</button>
      <button type="button" class="timeline-nav" data-idx="{prev_idx}">Prev</button>
      <div class="timeline-main">
        <div class="range-label">
          <span>{html.escape(pkl_path.name)}</span>
          <output id="idx-output">{idx}</output>
        </div>
        <input id="idx-range" name="idx" type="range" min="0" max="{len(transitions) - 1}" value="{idx}" step="1">
      </div>
      <input id="idx-number" type="number" min="0" max="{len(transitions) - 1}" value="{idx}">
      <button type="button" class="timeline-nav" data-idx="{next_idx}">Next</button>
      <button type="button" class="timeline-nav" data-idx="{len(transitions) - 1}">Last</button>
      <button type="submit">Go</button>
    </form>
    """
    return content


def html_doc(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
      padding-bottom: 96px;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    aside {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 18px;
      background: #ecf1f4;
      border-right: 1px solid var(--line);
    }}
    main {{ padding: 22px; max-width: 1500px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h2 {{ margin-top: 22px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
    form {{ display: grid; gap: 8px; }}
    input, select, button {{
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
    }}
    button {{ background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; }}
    .timeline {{
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: 68px 68px minmax(180px, 1fr) 96px 68px 68px 68px;
      gap: 10px;
      align-items: center;
      padding: 12px 18px;
      background: rgba(246, 247, 249, 0.96);
      border-top: 1px solid var(--line);
      box-shadow: 0 -8px 24px rgba(31, 41, 51, 0.10);
      backdrop-filter: blur(10px);
    }}
    .timeline-main {{
      display: grid;
      gap: 4px;
      min-width: 0;
    }}
    .timeline-main .range-label span {{
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      color: var(--muted);
    }}
    .timeline button,
    .timeline input[type="number"] {{
      height: 38px;
      padding: 7px 9px;
    }}
    .timeline-nav {{
      background: white;
      color: var(--accent);
      border-color: var(--line);
    }}
    .range-label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    output {{
      min-width: 72px;
      padding: 3px 8px;
      text-align: right;
      color: var(--accent);
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    input[type="range"] {{
      --progress: 0%;
      padding: 0;
      height: 28px;
      border: 0;
      border-radius: 0;
      background: transparent;
      accent-color: var(--accent);
      cursor: pointer;
    }}
    input[type="range"]::-webkit-slider-runnable-track {{
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(
        to right,
        var(--accent) 0,
        var(--accent) var(--progress),
        #cfd8df var(--progress),
        #cfd8df 100%
      );
    }}
    input[type="range"]::-webkit-slider-thumb {{
      margin-top: -5px;
    }}
    input[type="range"]::-moz-range-track {{
      height: 8px;
      border-radius: 999px;
      background: #cfd8df;
    }}
    input[type="range"]::-moz-range-progress {{
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
    }}
    nav {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 16px 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    nav a {{
      display: block;
      padding: 8px;
      text-align: center;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      margin: 8px 0 0;
      padding: 10px;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
    }}
    section {{
      margin-bottom: 16px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    details {{ margin: 8px 0; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
      padding: 10px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .image-frame {{
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-radius: 4px;
      border: 1px solid var(--line);
      background: #111827;
      contain: layout paint;
    }}
    img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      image-rendering: auto;
    }}
    figcaption {{ margin-top: 8px; color: var(--muted); word-break: break-all; }}
    .jumps {{ word-spacing: 8px; }}
    @media (max-width: 900px) {{
      body {{ display: block; padding-bottom: 152px; }}
      aside {{ position: static; height: auto; }}
      main {{ padding: 14px; }}
      .timeline {{
        grid-template-columns: 1fr 1fr 1fr 1fr;
      }}
      .timeline-main {{
        grid-column: 1 / -1;
        order: -1;
      }}
      .timeline input[type="number"] {{
        grid-column: span 2;
      }}
    }}
  </style>
</head>
<body>
{body}
<script id="browser-script">
  let submitTimer = null;
  let activeController = null;

  function currentFile() {{
    return document.querySelector(".timeline input[name='file']")?.value || "0";
  }}

  function setRangeProgress(range) {{
    if (!range) return;
    const min = Number(range.min);
    const max = Number(range.max);
    const value = Number(range.value || 0);
    const percent = max > min ? ((value - min) / (max - min)) * 100 : 0;
    range.style.setProperty("--progress", `${{percent}}%`);
  }}

  function syncIndex(value, submit) {{
    const form = document.querySelector(".timeline");
    const range = document.getElementById("idx-range");
    const number = document.getElementById("idx-number");
    const output = document.getElementById("idx-output");
    if (!form || !range || !number || !output) return;
    const min = Number(range.min);
    const max = Number(range.max);
    const next = Math.max(min, Math.min(max, Number(value || 0)));
    range.value = String(next);
    number.value = String(next);
    output.value = String(next);
    output.textContent = String(next);
    setRangeProgress(range);
    if (submit) {{
      window.clearTimeout(submitTimer);
      submitTimer = window.setTimeout(() => goToIndex(next), 180);
    }}
  }}

  async function goToIndex(idx, pushHistory = true) {{
    const file = currentFile();
    const url = `/?file=${{encodeURIComponent(file)}}&idx=${{encodeURIComponent(idx)}}`;
    const fragmentUrl = `/fragment?file=${{encodeURIComponent(file)}}&idx=${{encodeURIComponent(idx)}}`;
    if (activeController) activeController.abort();
    activeController = new AbortController();
    document.body.classList.add("loading-step");
    try {{
      const response = await fetch(fragmentUrl, {{ signal: activeController.signal }});
      if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
      const html = await response.text();
      await preloadFragmentImages(html);
      const scriptHtml = document.getElementById("browser-script")?.outerHTML || "";
      document.body.innerHTML = html + scriptHtml;
      if (pushHistory) history.pushState({{ file, idx }}, "", url);
      bindControls();
    }} catch (error) {{
      if (error.name !== "AbortError") {{
        window.location.href = url;
      }}
    }} finally {{
      document.body.classList.remove("loading-step");
    }}
  }}

  function preloadFragmentImages(fragmentHtml) {{
    const doc = new DOMParser().parseFromString(fragmentHtml, "text/html");
    const srcs = Array.from(doc.querySelectorAll("img"))
      .map((img) => img.getAttribute("src"))
      .filter(Boolean);
    if (srcs.length === 0) return Promise.resolve();
    return Promise.all(
      srcs.map((src) => new Promise((resolve) => {{
        const image = new Image();
        image.onload = resolve;
        image.onerror = resolve;
        image.src = src;
      }}))
    );
  }}

  function bindControls() {{
    const form = document.querySelector(".timeline");
    const range = document.getElementById("idx-range");
    const number = document.getElementById("idx-number");
    const output = document.getElementById("idx-output");
    const navButtons = document.querySelectorAll(".timeline-nav");
    const sidebarLinks = document.querySelectorAll("aside a[href^='/?']");

    range?.addEventListener("input", () => syncIndex(range.value, false));
    range?.addEventListener("change", () => syncIndex(range.value, true));
    number?.addEventListener("input", () => syncIndex(number.value, false));
    number?.addEventListener("change", () => syncIndex(number.value, true));
    navButtons.forEach((button) => {{
      button.addEventListener("click", () => syncIndex(button.dataset.idx, true));
    }});
    sidebarLinks.forEach((link) => {{
      link.addEventListener("click", (event) => {{
        event.preventDefault();
        const params = new URL(link.href).searchParams;
        syncIndex(params.get("idx") || 0, true);
      }});
    }});
    form?.addEventListener("submit", (event) => {{
      event.preventDefault();
      goToIndex(number?.value || range?.value || 0);
    }});
    if (output && range) {{
      output.value = range.value;
      output.textContent = range.value;
    }}
    setRangeProgress(range);
  }}

  window.addEventListener("popstate", () => {{
    const params = new URLSearchParams(window.location.search);
    goToIndex(params.get("idx") || 0, false);
  }});

  bindControls();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer_dir", help="Directory containing transitions_*.pkl files, or one pkl file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cache_size", type=int, default=3)
    args = parser.parse_args()

    browser = ReplayBufferBrowser(Path(args.buffer_dir), cache_size=args.cache_size)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(browser))
    url = f"http://{args.host}:{args.port}"
    print(f"Replay buffer browser: {url}")
    print(f"Browsing: {browser.buffer_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
