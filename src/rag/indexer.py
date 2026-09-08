import re
from pathlib import Path
from typing import List, Dict, Any

class FMEAChunk:
    def __init__(self, doc_id: str, section_title: str, content: str, failure_classes: List[str], tool_chamber: str):
        self.doc_id = doc_id
        self.section_title = section_title
        self.content = content.strip()
        self.failure_classes = failure_classes
        self.tool_chamber = tool_chamber

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "section_title": self.section_title,
            "content": self.content,
            "failure_classes": self.failure_classes,
            "tool_chamber": self.tool_chamber
        }

class FMEAIndexer:
    """
    Parses SEMI-E10 Markdown FMEA playbooks into semantically coherent retrieval chunks.
    """
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.chunks: List[FMEAChunk] = []

    def load_and_chunk_corpus(self) -> List[FMEAChunk]:
        self.chunks = []
        doc_order = [
            "300mm_rie_plasma_etch_fmea.md",
            "300mm_photolithography_fmea.md",
            "300mm_cmp_planarization_fmea.md"
        ]
        ordered_files = [self.corpus_dir / f for f in doc_order if (self.corpus_dir / f).exists()]
        remaining = sorted([p for p in self.corpus_dir.glob("*.md") if p not in ordered_files])
        md_files = ordered_files + remaining
        if not md_files:
            raise FileNotFoundError(f"No FMEA markdown files found in {self.corpus_dir}")

        for filepath in md_files:
            text = filepath.read_text(encoding="utf-8")
            self._chunk_document(text, filepath.name)

        return self.chunks

    def _chunk_document(self, text: str, filename: str) -> None:
        doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([A-Za-z0-9\-_]+)", text)
        doc_id = doc_id_match.group(1) if doc_id_match else filename.replace(".md", "")

        chamber_match = re.search(r"\*\*Target (?:Chamber|Station):\*\*\s*(.+)", text)
        tool_chamber = chamber_match.group(1).strip() if chamber_match else "General"

        classes_match = re.search(r"\*\*Classification Coverage:\*\*\s*(.+)", text)
        failure_classes = [c.strip() for c in classes_match.group(1).split(",")] if classes_match else []

        # Split document on H2 sections (## )
        sections = re.split(r"(?m)^##\s+", text)
        for sec in sections:
            if not sec.strip() or sec.startswith("# "):
                continue
            lines = sec.split("\n")
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            chunk = FMEAChunk(
                doc_id=doc_id,
                section_title=title,
                content=body,
                failure_classes=failure_classes,
                tool_chamber=tool_chamber
            )
            self.chunks.append(chunk)
