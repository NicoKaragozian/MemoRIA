import json
from pathlib import Path

from scripts.anonymize import anonymize


def _extract_pdf(filepath: str) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_docx(filepath: str) -> str:
    from docx import Document
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    # Incluir párrafos cortos (>20 chars) para no perder títulos y secciones
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    chunks = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            tail = current[-overlap:] if overlap < len(current) else current[:]
            current = tail + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def parse_academic_folder(folder: str, min_chunk_len: int = 200) -> list[dict]:
    """
    Procesa todos los PDFs y DOCX en la carpeta.
    Aplica anonimización de PII antes de retornar.
    """
    folder_path = Path(folder)
    files = list(folder_path.glob("**/*.pdf")) + list(folder_path.glob("**/*.docx"))

    examples = []
    failed = 0

    for filepath in files:
        print(f"Procesando: {filepath.name}")
        try:
            if filepath.suffix == ".pdf":
                text = _extract_pdf(str(filepath))
            else:
                text = _extract_docx(str(filepath))

            if not text or len(text) < 500:
                print(f"  ⚠ Texto muy corto, saltando")
                continue

            chunks = [c for c in split_into_chunks(text) if len(c) >= min_chunk_len]
            for chunk in chunks:
                examples.append({
                    "text": anonymize(chunk),
                    "source": filepath.name,
                    "register": "academic",
                })
            print(f"  → {len(chunks)} chunks")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    if failed:
        print(f"\n⚠ {failed}/{len(files)} archivos fallaron")

    return examples


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw/academic"
    examples = parse_academic_folder(folder)
    print(f"\nTotal: {len(examples)} chunks académicos")
    output = "academic_parsed.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Guardado en {output}")
