#!/usr/bin/env python3
"""Build the focused WALL-OSS-0.5 IREE implementation article.

One Markdown source produces a self-contained HTML article, a Zhihu Markdown
copy, and a Downloads package containing the same evidence images.
"""
from __future__ import annotations

import base64
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / 'wallx_iree_implementation.md'
ASSETS = DOCS / 'mlir_tutorial' / 'assets' / 'katex'
DOWNLOADS = Path('/mnt/c/Users/Admin1/Downloads')
PKG = DOWNLOADS / 'wallx-iree-implementation-zhihu'


def inline(text: str) -> str:
    maths: list[str] = []
    codes: list[str] = []
    links: list[tuple[str, str]] = []

    def keep_math(match):
        maths.append(match.group(0))
        return f'@@M{len(maths) - 1}@@'

    text = re.sub(r'\$\$[\s\S]+?\$\$|\$[^$\n]+?\$', keep_math, text)

    def keep_code(match):
        codes.append(match.group(1))
        return f'@@C{len(codes) - 1}@@'

    text = re.sub(r'`([^`\n]+)`', keep_code, text)

    def keep_link(match):
        links.append((match.group(1), match.group(2)))
        return f'@@L{len(links) - 1}@@'

    text = re.sub(r'\[([^\]\n]+)\]\(([^\s)]+)\)', keep_link, text)
    text = html.escape(text, quote=False)
    text = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', text)
    for i, raw in enumerate(codes):
        text = text.replace(f'@@C{i}@@',
                            f'<code>{html.escape(raw, quote=False)}</code>')
    for i, (label, url) in enumerate(links):
        attrs = ' target="_blank" rel="noopener"' if url.startswith('http') else ''
        text = text.replace(
            f'@@L{i}@@',
            f'<a href="{html.escape(url, quote=True)}"{attrs}>'
            f'{html.escape(label, quote=False)}</a>')
    for i, raw in enumerate(maths):
        text = text.replace(f'@@M{i}@@', html.escape(raw, quote=False))
    return text


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return [cell.strip() for cell in line.split('|')]


def is_separator(line: str) -> bool:
    cells = split_row(line)
    return bool(cells) and all(
        re.fullmatch(r':?-{2,}:?', cell.replace(' ', '')) for cell in cells)


def image_tag(alt: str, source: str) -> str:
    path = (DOCS / source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'article image missing: {path}')
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return (f'<figure><img src="data:{mime};base64,{data}" '
            f'alt="{html.escape(alt, quote=True)}">'
            f'<figcaption>{inline(alt)}</figcaption></figure>')


def native_log_path(relative: str) -> Path:
    path = (DOCS / relative).resolve()
    if DOCS.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f'native log missing: {path}')
    return path


NATIVE_LOG_RANGES = {
    'implementation_evidence/native_logs/full_flow_export.console.log': [(22, 32), (39, 50)],
    'implementation_evidence/native_logs/full_flow_iree_compile.console.log': [(17, 27), (71, 77), (24690, 24697)],
    'implementation_evidence/native_logs/expanded_serving.console.log': [(415, 424), (706, 716)],
    'implementation_evidence/native_logs/full_flow_verify.console.log': [(35, 48), (683, 704), (713, 730)],
    'implementation_evidence/native_logs/staged_verify.console.log': [(37, 39), (50, 52), (83, 84), (91, 92)],
    'implementation_evidence/native_logs/weights.console.log': [(18, 22), (29, 29), (35, 38), (43, 49), (64, 78), (95, 96)],
}


def native_log_excerpt(relative: str) -> str:
    path = native_log_path(relative)
    lines = path.read_text(encoding='utf-8').splitlines()
    if relative.endswith('backend_compare.console.log'):
        blocks = {}
        index = 0
        while index < len(lines):
            if lines[index].strip() == '{' and index + 1 < len(lines) \
                    and '"backend":' in lines[index + 1]:
                backend = lines[index + 1].split('"backend":', 1)[1].split('"')[1]
                end = index + 1
                while end < len(lines) and lines[end].strip() != '}':
                    end += 1
                blocks[backend] = lines[index:end + 1]
                index = end
            index += 1
        selected = []
        for backend in ('official', 'torch-compile', 'staged', 'expanded'):
            if backend not in blocks:
                raise ValueError(f'missing native backend summary: {backend}')
            selected.extend(blocks[backend])
        return '\n'.join(selected)
    ranges = NATIVE_LOG_RANGES.get(relative)
    if ranges is None:
        return '\n'.join(lines)
    selected = []
    for start, end in ranges:
        if end > len(lines):
            raise ValueError(f'native log excerpt outside file: {relative} {start}-{end}')
        selected.extend(lines[start - 1:end])
    return '\n'.join(selected)


def expand_native_logs(source: str) -> str:
    pattern = re.compile(r'^::native-log\s+(\S+)\s*$', re.M)

    def replace(match):
        relative = match.group(1)
        path = native_log_path(relative)
        raw = native_log_excerpt(relative)
        return (f'**关键原生日志节选：`{relative}`（完整文件 {path.stat().st_size} bytes，随 evidence 保存）**\n\n'
                f'```text\n{raw}\n```')
    return pattern.sub(replace, source)


def render_markdown(source: str):
    lines = source.splitlines()
    out: list[str] = []
    headings: list[tuple[int, str, str]] = []
    title = ''
    subtitle = ''
    paragraph: list[str] = []
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            out.append('<p>' + inline(' '.join(x.strip() for x in paragraph)) + '</p>')
            paragraph = []

    while i < len(lines):
        line = lines[i]
        native_log = re.fullmatch(r'::native-log\s+(\S+)', line.strip())
        if native_log:
            flush_paragraph()
            relative = native_log.group(1)
            path = native_log_path(relative)
            raw = native_log_excerpt(relative)
            out.append(
                f'<details class="native-log"><summary>关键原生日志节选：'
                f'{html.escape(relative)}（完整文件 {path.stat().st_size} bytes，随 evidence 保存）</summary>'
                f'<pre><code>{html.escape(raw, quote=False)}</code></pre></details>')
            i += 1
            continue
        if line.startswith('```'):
            flush_paragraph()
            language = line[3:].strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError('unclosed code fence')
            cls = (f' class="language-{html.escape(language, quote=True)}"'
                   if language else '')
            out.append(f'<pre><code{cls}>' +
                       html.escape('\n'.join(code), quote=False) +
                       '</code></pre>')
            i += 1
            continue
        if not line.strip():
            flush_paragraph()
            i += 1
            continue
        image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if image:
            flush_paragraph()
            out.append(image_tag(image.group(1), image.group(2)))
            i += 1
            continue
        if line.startswith('# '):
            flush_paragraph()
            title = line[2:].strip()
            i += 1
            continue
        if (line.startswith('*') and line.endswith('*')
                and not line.startswith('**') and not subtitle):
            flush_paragraph()
            subtitle = line.strip('*').strip()
            i += 1
            continue
        heading = re.match(r'^(#{2,3})\s+(.+?)\s*$', line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            identifier = f'sec-{len(headings) + 1}'
            headings.append((level, text, identifier))
            out.append(f'<h{level} id="{identifier}">{inline(text)}</h{level}>')
            i += 1
            continue
        if line.strip() == '---':
            flush_paragraph()
            out.append('<hr>')
            i += 1
            continue
        if (line.strip().startswith('|') and i + 1 < len(lines)
                and is_separator(lines[i + 1])):
            flush_paragraph()
            headers = split_row(line)
            aligns = ['right' if cell.replace(' ', '').endswith(':') else 'left'
                      for cell in split_row(lines[i + 1])]
            out.append('<div class="table-wrap"><table><thead><tr>' + ''.join(
                f'<th style="text-align:{aligns[j] if j < len(aligns) else "left"}">'
                f'{inline(cell)}</th>' for j, cell in enumerate(headers)) +
                '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = split_row(lines[i])
                out.append('<tr>' + ''.join(
                    f'<td style="text-align:{aligns[j] if j < len(aligns) else "left"}">'
                    f'{inline(cell)}</td>' for j, cell in enumerate(cells)) + '</tr>')
                i += 1
            out.append('</tbody></table></div>')
            continue
        if line.startswith('- '):
            flush_paragraph()
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(lines[i][2:])
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inline(item)}</li>' for item in items) + '</ul>')
            continue
        if re.match(r'^\d+\.\s+', line):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i]):
                items.append(re.sub(r'^\d+\.\s+', '', lines[i]))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{inline(item)}</li>' for item in items) + '</ol>')
            continue
        if line.startswith('>'):
            flush_paragraph()
            quote = []
            while i < len(lines) and lines[i].startswith('>'):
                quote.append(lines[i][1:].lstrip())
                i += 1
            out.append('<blockquote>' + '<br>'.join(inline(x) for x in quote) + '</blockquote>')
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return '\n'.join(out), title, subtitle, headings


def font_css() -> str:
    specs = [
        ('KaTeX_Main', 'KaTeX_Main-Regular.woff2', 400, 'normal'),
        ('KaTeX_Main', 'KaTeX_Main-Italic.woff2', 400, 'italic'),
        ('KaTeX_Main', 'KaTeX_Main-Bold.woff2', 700, 'normal'),
        ('KaTeX_Math', 'KaTeX_Math-Italic.woff2', 400, 'italic'),
        ('KaTeX_Size1', 'KaTeX_Size1-Regular.woff2', 400, 'normal'),
        ('KaTeX_Size2', 'KaTeX_Size2-Regular.woff2', 400, 'normal'),
        ('KaTeX_Size3', 'KaTeX_Size3-Regular.woff2', 400, 'normal'),
        ('KaTeX_Size4', 'KaTeX_Size4-Regular.woff2', 400, 'normal'),
    ]
    result = []
    for family, filename, weight, style in specs:
        data = base64.b64encode((ASSETS / filename).read_bytes()).decode('ascii')
        result.append("@font-face{font-family:'%s';src:url(data:font/woff2;base64,%s) format('woff2');font-weight:%s;font-style:%s;}" %
                      (family, data, weight, style))
    return '\n'.join(result)


STYLE = r'''
:root{--ink:#20252b;--muted:#657180;--accent:#20549b;--line:#dce3eb;--bg:#f7f9fc;--code:#f3f5f8}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif;line-height:1.82;font-size:16px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.layout{max-width:1420px;margin:0 auto;display:grid;grid-template-columns:285px minmax(0,1fr);gap:34px;padding:28px 26px 60px}
.toc{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 16px;font-size:13px;line-height:1.55}
.toc h2{font-size:16px;margin:0 0 12px;color:#183b64}.toc a{display:block;color:#46566b;padding:3px 0}.toc a.l3{padding-left:15px;font-size:12px;color:#728094}
main{min-width:0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:44px 52px 56px;box-shadow:0 4px 18px #20314a08}
h1{font-size:31px;line-height:1.35;color:#143b66;margin:0 0 14px;letter-spacing:-.2px}h2{font-size:24px;color:#174d80;border-bottom:1px solid var(--line);padding-bottom:8px;margin:40px 0 18px;line-height:1.45}h3{font-size:19px;color:#245a83;margin:28px 0 12px;line-height:1.5}p{margin:13px 0}strong{color:#152f4c;font-weight:650}em{color:#53677a}
.subtitle{color:var(--muted);font-size:16px;margin:0 0 22px;line-height:1.65}.meta{color:var(--muted);font-size:13px;border-top:1px solid var(--line);margin-top:36px;padding-top:16px}
pre{background:var(--code);border:1px solid #e0e6ee;border-left:4px solid #7c9bc0;border-radius:8px;padding:15px 17px;overflow:auto;line-height:1.58;font-size:12.5px;margin:18px 0}code{font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;font-size:.91em;background:#eef2f6;border-radius:4px;padding:1px 4px}pre code{background:transparent;padding:0;font-size:inherit}p code,li code,td code,th code,h2 code,h3 code,figcaption code{overflow-wrap:anywhere;word-break:break-word}
.table-wrap{overflow-x:auto;margin:17px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;line-height:1.55}th,td{border:1px solid var(--line);padding:7px 9px;vertical-align:top}th{background:#edf3f9;color:#183e61;font-weight:650}tr:nth-child(even) td{background:#fbfcfe}
details.native-log{margin:20px 0;border:1px solid #d5dfea;border-radius:9px;background:#f7f9fc}details.native-log summary{cursor:pointer;padding:12px 15px;color:#245a83;font-weight:650;font-size:13px;overflow-wrap:anywhere;word-break:break-word}details.native-log pre{margin:0;max-height:560px;overflow:auto;border:0;border-top:1px solid #d5dfea;border-radius:0 0 8px 8px;font-size:11px;white-space:pre}
ul,ol{padding-left:26px}li{margin:6px 0}blockquote{border-left:4px solid #9ab4d2;background:#f2f6fb;padding:12px 18px;color:#44576b;margin:18px 0}hr{border:0;border-top:1px solid var(--line);margin:30px 0}.katex-display{overflow-x:auto;padding:8px 0}.katex{font-size:1.02em}
figure{margin:24px 0;padding:12px;background:#f8fafc;border:1px solid var(--line);border-radius:10px}figure img{display:block;width:100%;height:auto;border-radius:5px;background:#300a24}figcaption{color:var(--muted);font-size:13px;line-height:1.55;margin-top:9px;text-align:center}
@media(max-width:1000px){.layout{display:block;padding:12px}.toc{position:relative;max-height:none;margin-bottom:16px}main{padding:26px 22px}h1{font-size:26px}h2{font-size:22px}}
@media print{body{background:#fff}.layout{display:block;max-width:none;padding:0}.toc{display:none}main{border:0;box-shadow:none;padding:0}h2{break-after:avoid}pre,table,blockquote,figure{break-inside:avoid}a{color:inherit;text-decoration:none}}
'''


def build_html(source: str) -> str:
    body, title, subtitle, headings = render_markdown(source)
    toc = ['<div class="toc"><h2>文章目录</h2>']
    for level, text, identifier in headings:
        cls = ' class="l3"' if level == 3 else ''
        toc.append(f'<a{cls} href="#{identifier}">{html.escape(text)}</a>')
    toc.append('</div>')
    css = re.sub(r'@font-face\s*\{[^}]*\}', '',
                 (ASSETS / 'katex.min.css').read_text(encoding='utf-8'))
    katex_js = (ASSETS / 'katex.min.js').read_text(encoding='utf-8')
    auto_js = (ASSETS / 'auto-render.min.js').read_text(encoding='utf-8')
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="为具身 Flow/VLA 模型接入 IREE：通用模型契约、后端分层、真实 IR 与原生日志、参数外置、设备 ABI、官方 serving 和 torch.compile 对照。">
<style>{STYLE}{css}{font_css()}</style></head>
<body><div class="layout">{''.join(toc)}<main><h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p>{body}
<p class="meta">本文与 docs/iree_wallx/README.md、scripts/iree_serving.sh 使用同一实现口径。真实终端截图内嵌于 HTML；完整原生日志、逐次结果和捕获元数据随 evidence 保存。公式由内嵌 KaTeX 渲染。</p>
</main></div><script>{katex_js}</script><script>{auto_js}</script>
<script>document.addEventListener('DOMContentLoaded',function(){{
  if (typeof renderMathInElement === 'function') {{
    renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}], ignoredTags:['script','noscript','style','textarea','pre','code'], throwOnError:false}});
  }}
}});</script></body></html>'''


def write_outputs(source: str):
    markdown_source = expand_native_logs(source)
    md_path = DOCS / 'wallx_iree_implementation.zhihu.md'
    html_path = DOCS / 'wallx_iree_implementation.html'
    md_path.write_text(markdown_source, encoding='utf-8')
    html_text = build_html(source)
    html_path.write_text(html_text, encoding='utf-8')

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    (DOWNLOADS / 'wallx-iree-implementation.html').write_text(html_text, encoding='utf-8')
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / 'wallx-iree-implementation.zhihu.md').write_text(markdown_source, encoding='utf-8')
    (PKG / 'wallx-iree-implementation.html').write_text(html_text, encoding='utf-8')
    evidence_out = PKG / 'implementation_evidence'
    evidence_out.mkdir(exist_ok=True)
    for relative in re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', source, re.M):
        image = (DOCS / relative).resolve()
        shutil.copy2(image, evidence_out / image.name)
        vector = image.with_suffix('.svg')
        if vector.is_file():
            shutil.copy2(vector, evidence_out / vector.name)
    for image in sorted((DOCS / 'implementation_evidence').glob('native_*.png')):
        shutil.copy2(image, evidence_out / image.name)
    native_logs = DOCS / 'implementation_evidence' / 'native_logs'
    if native_logs.is_dir():
        shutil.copytree(native_logs, evidence_out / 'native_logs', dirs_exist_ok=True)
    comparison = DOCS / 'implementation_evidence' / 'native_backend_comparison.json'
    if comparison.is_file():
        shutil.copy2(comparison, evidence_out / comparison.name)
    metadata = DOCS / 'implementation_evidence' / 'html_validation.json'
    if metadata.is_file():
        shutil.copy2(metadata, evidence_out / metadata.name)
    paper_manifest = DOCS / 'implementation_evidence' / 'paper_figures.json'
    if paper_manifest.is_file():
        shutil.copy2(paper_manifest, evidence_out / paper_manifest.name)
    model_spec = DOCS / 'implementation_evidence' / 'model_visual_spec.json'
    if model_spec.is_file():
        shutil.copy2(model_spec, evidence_out / model_spec.name)
    dot_source = DOCS / 'implementation_evidence' / 'dot'
    if dot_source.is_dir():
        shutil.copytree(dot_source, evidence_out / 'dot', dirs_exist_ok=True)
    ir_source = DOCS / 'implementation_evidence' / 'ir' / 'full_flow'
    if ir_source.is_dir():
        ir_out = evidence_out / 'ir' / 'full_flow'
        ir_out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((ir_source / 'manifest.json').read_text(encoding='utf-8'))
        prefixes = ('configuration', 'flow', 'input', 'stream', 'torch', 'vm')
        selected = {name: value for name, value in manifest.get('files', {}).items()
                    if name.startswith(prefixes)}
        for name in selected:
            shutil.copy2(ir_source / name, ir_out / name)
        manifest['files'] = selected
        (ir_out / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')

    package_readme = '''# 具身 Flow/VLA 模型的 IREE 后端发布包

- `wallx-iree-implementation.html`：自包含 HTML，内嵌图像、KaTeX、字体和关键日志节选，完整原生日志随 evidence 保存。
- `wallx-iree-implementation.zhihu.md`：知乎导入用 Markdown，包含同样的关键日志节选。
- `implementation_evidence/`：结构图、真实终端截图、native logs、四后端结果数组、`full_flow` 阶段 IR 和验收元数据。
- `wallx_iree_implementation.md`：展开原生日志后的同源文章。
- `build_wallx_iree_implementation.py`：同源重建脚本。
- `backend-guide.md`：环境、编译、serving、日志和四后端对照指南。
- `iree_serving.sh`：保留原生服务器输出的官方 serving 启动脚本。
- `draw_model_structure.py`：兼容入口，调用统一结构化渲染器。
- `render_model_graphs.py`：以 Torchview + Graphviz 生成全部算子图、后端图和 mask 图。
- `walloss_graph_trial.py`：Torchview 结构化模型定义与渲染入口。
- `validate_layout.py`：节点、文字、裁切和语义布局校验。
- `model_visual_spec.json`：六张 Torchview 图的布局与语义验证记录。
- `dot/`：全部图的 Graphviz DOT 源文件。
- `draw_backend_architecture.py`：eager、torch.compile、IREE 并列后端架构图生成脚本。
- `collect_full_flow_ir_evidence.py`：当前 `full_flow` 阶段 IR 摘录脚本。

HTML 无需外部图片或 CDN。Markdown 与 `implementation_evidence/` 保持相对路径，
在知乎编辑器中导入时请同时保留该目录。文章以具身 Flow/VLA 的通用后端契约为主线，
以 Wall-OSS-0.5 为案例展开相关 pass、运行证据和 torch.compile 对照。
'''
    (PKG / 'README.md').write_text(package_readme, encoding='utf-8')
    (PKG / SOURCE.name).write_text(markdown_source, encoding='utf-8')
    shutil.copy2(Path(__file__), PKG / Path(__file__).name)
    shutil.copy2(DOCS / 'README.md', PKG / 'backend-guide.md')
    shutil.copy2(DOCS.parents[1] / 'scripts' / 'iree_serving.sh',
                 PKG / 'iree_serving.sh')
    shutil.copy2(DOCS / 'draw_model_structure.py', PKG / 'draw_model_structure.py')
    shutil.copy2(DOCS / 'draw_backend_architecture.py',
                 PKG / 'draw_backend_architecture.py')
    shutil.copy2(DOCS / 'draw_mot_layer_structure.py',
                 PKG / 'draw_mot_layer_structure.py')
    shutil.copy2(DOCS / 'draw_attention_mask_structure.py',
                 PKG / 'draw_attention_mask_structure.py')
    shutil.copy2(DOCS / 'render_model_graphs.py', PKG / 'render_model_graphs.py')
    shutil.copy2(DOCS / 'walloss_graph_trial.py', PKG / 'walloss_graph_trial.py')
    shutil.copy2(DOCS / 'validate_layout.py', PKG / 'validate_layout.py')
    shutil.copy2(DOCS / 'model_visual_spec.json', PKG / 'model_visual_spec.json')
    shutil.copy2(DOCS / 'collect_full_flow_ir_evidence.py',
                 PKG / 'collect_full_flow_ir_evidence.py')
    with zipfile.ZipFile(DOWNLOADS / 'wallx-iree-implementation-zhihu.zip', 'w',
                         zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(PKG.rglob('*')):
            if file.is_file():
                archive.write(file, file.relative_to(PKG))
    return md_path, html_path


def main():
    source = SOURCE.read_text(encoding='utf-8')
    fences = len(re.findall(r'^```', source, re.M))
    if fences % 2:
        raise ValueError('unclosed code fence')
    images = re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', source, re.M)
    if not images:
        raise ValueError('article has no evidence images')
    md_path, html_path = write_outputs(source)
    print(f'source: {SOURCE} ({len(source.encode("utf-8"))} bytes)')
    print(f'evidence images: {len(images)}')
    print(f'HTML: {html_path} ({html_path.stat().st_size} bytes)')
    print(f'Markdown: {md_path} ({md_path.stat().st_size} bytes)')
    print(f'package: {PKG}')
    print(f'zip: {DOWNLOADS / "wallx-iree-implementation-zhihu.zip"}')


if __name__ == '__main__':
    main()
