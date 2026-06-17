"""Load markdown files (Notion/Obsidian exports) into InputData format."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from util.io_schemas import InputData, Section, SourceNode


def load_markdown_dir(path: Path, *, recursive: bool = True) -> InputData:
    """
    Load markdown files into InputData.

    Args:
        path: Can be:
            - A single .md file
            - A directory containing .md files
            - A JSON file containing a list of .md file paths
        recursive: Whether to search subdirectories (default: True)

    Returns:
        InputData with one SourceNode per markdown file
    """
    path = Path(path)

    # Case 1: Single .md file
    if path.is_file() and path.suffix.lower() == ".md":
        source_node = _md_file_to_source_node(path, idx=0)
        if source_node is None:
            return InputData(source_nodes=[])
        return InputData(source_nodes=[source_node])

    # Case 2: JSON file containing list of .md paths
    if path.is_file() and path.suffix.lower() == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                md_paths = json.load(f)
            if isinstance(md_paths, list) and all(
                isinstance(p, str) and p.endswith(".md") for p in md_paths
            ):
                source_nodes = []
                for idx, md_path_str in enumerate(md_paths):
                    md_path = Path(md_path_str)
                    if not md_path.is_absolute():
                        md_path = path.parent / md_path
                    if md_path.exists():
                        node = _md_file_to_source_node(md_path, idx=idx)
                        if node:
                            source_nodes.append(node)
                return InputData(source_nodes=source_nodes)
        except (json.JSONDecodeError, IOError):
            pass

    # Case 3: Directory
    if path.is_dir():
        pattern = "**/*.md" if recursive else "*.md"
        md_files = sorted(path.glob(pattern))
        source_nodes = []
        for idx, md_file in enumerate(md_files):
            node = _md_file_to_source_node(md_file, idx=idx)
            if node:
                source_nodes.append(node)
        return InputData(source_nodes=source_nodes)

    # Invalid path or no files found
    return InputData(source_nodes=[])


def _md_file_to_source_node(path: Path, idx: int) -> Optional[SourceNode]:
    """
    Convert one .md file to one SourceNode.

    Args:
        path: Path to markdown file
        idx: Index for ID generation

    Returns:
        SourceNode or None if file is empty/invalid
    """
    try:
        # Read file with error replacement for encoding issues
        text = path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return None

    if not text.strip():
        return None

    # Extract YAML front-matter
    metadata, body = _extract_front_matter(text)

    if not body.strip():
        return None

    # Extract title from front-matter or filename
    title = metadata.get("title", path.stem)

    # Extract timestamps
    create_time = None
    for key in ["created", "created_at", "date"]:
        if key in metadata:
            create_time = _parse_timestamp(metadata[key])
            if create_time:
                break
    if create_time is None:
        try:
            create_time = int(path.stat().st_ctime)
        except (OSError, AttributeError):
            create_time = None

    update_time = None
    for key in ["updated", "modified", "updated_at"]:
        if key in metadata:
            update_time = _parse_timestamp(metadata[key])
            if update_time:
                break
    if update_time is None:
        try:
            update_time = int(path.stat().st_mtime)
        except (OSError, AttributeError):
            update_time = None

    # Split body into sections
    section_tuples = _split_into_sections(body)

    if not section_tuples:
        return None

    # Convert to Section objects
    sections = []
    for i, (heading_title, section_content) in enumerate(section_tuples):
        section = Section(
            id=f"md_{idx}_sec_{i}",
            content=section_content,
            role=None,
            section_title=heading_title if heading_title not in ["(intro)", "(note)"] else None,
        )
        sections.append(section)

    # Build SourceNode ID (sanitize filename for ID)
    safe_name = re.sub(r'[^\w\-]', '_', path.stem)[:40]
    source_id = f"md_{idx}_{safe_name}"

    return SourceNode(
        id=source_id,
        title=title,
        sections=sections,
        source_type="markdown",
        create_time=create_time,
        update_time=update_time,
    )


def _parse_timestamp(raw: str) -> Optional[int]:
    """
    Parse date string to Unix timestamp int.

    Args:
        raw: Date string in various formats

    Returns:
        Unix timestamp or None if parsing fails
    """
    if not isinstance(raw, str):
        # Handle case where it's already an int
        if isinstance(raw, int):
            return raw
        return None

    # Only use first 19 characters for parsing
    raw = raw[:19].strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            continue

    return None


def _extract_front_matter(text: str) -> Tuple[dict, str]:
    """
    Extract YAML front-matter from markdown text.

    Args:
        text: Full markdown text

    Returns:
        Tuple of (metadata_dict, body_without_front_matter)
    """
    # Match front-matter: ^---\n...\n---\n
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(pattern, text, re.DOTALL)

    if not match:
        return {}, text

    front_matter = match.group(1)
    body = text[match.end():]

    # Parse front-matter as simple key: value pairs
    metadata = {}
    for line in front_matter.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        key, _, value = line.partition(':')
        key = key.strip()
        value = value.strip()

        # Handle lists [item1, item2]
        if value.startswith('[') and value.endswith(']'):
            value = [item.strip() for item in value[1:-1].split(',')]
        # Remove quotes
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

        metadata[key] = value

    return metadata, body


def _split_into_sections(body: str) -> List[Tuple[str, str]]:
    """
    Split markdown body by headings into sections.

    Args:
        body: Markdown text without front-matter

    Returns:
        List of (heading_title, section_content) tuples
    """
    # Find all headings (# ## ###)
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(body))

    if not matches:
        # No headings - treat entire body as single section
        cleaned = _clean_markdown(body)
        if cleaned and len(cleaned) >= 10:
            return [("(note)", cleaned)]
        return []

    sections = []

    # Content before first heading
    if matches[0].start() > 0:
        intro_content = body[:matches[0].start()]
        cleaned = _clean_markdown(intro_content)
        if cleaned and len(cleaned) >= 10:
            sections.append(("(intro)", cleaned))

    # Process each heading and its content
    for i, match in enumerate(matches):
        heading_title = match.group(2).strip()

        # Get content until next heading (or end of text)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_content = body[start:end]

        cleaned = _clean_markdown(section_content)
        if cleaned and len(cleaned) >= 10:
            sections.append((heading_title, cleaned))

    return sections


def _clean_markdown(text: str) -> str:
    """
    Remove markdown syntax that adds noise to embeddings.

    Args:
        text: Raw markdown text

    Returns:
        Cleaned text suitable for embedding
    """
    # Remove Obsidian callouts: ^> [!note] or similar
    text = re.sub(r'^>\s*\[!.*?\].*$', '', text, flags=re.MULTILINE)

    # Remove images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove fenced code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r'`[^`]+`', ' ', text)

    # Convert wikilinks: [[Link]] or [[Link|alias]] -> keep link/alias
    def replace_wikilink(match):
        content = match.group(1)
        if '|' in content:
            return content.split('|', 1)[1]  # Use alias
        return content  # Use link text
    text = re.sub(r'\[\[([^\]]+)\]\]', replace_wikilink, text)

    # Convert markdown links: [text](url) -> keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove horizontal rules
    text = re.sub(r'^[\-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove blockquote markers, keep content
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_

    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
