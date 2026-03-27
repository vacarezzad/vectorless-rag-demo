import json
import re
from typing import List, Dict, Tuple, Optional

from core.llm_client import LLMClient


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Extract JSON from an LLM response, stripping markdown code fences."""
    text = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL)
    text = text.strip()
    # Try to find a JSON array or object even if there is surrounding text
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


# ── Step 1: Tree generation ───────────────────────────────────────────────────

def generate_tree(pages: List[Dict], llm: LLMClient) -> List[Dict]:
    """
    Convert document pages into a hierarchical tree structure.

    The LLM reads a compact preview of each page and produces a JSON array
    of tree nodes — each with title, summary, page range, and hierarchy info.
    This replaces chunking + embedding entirely.
    """
    MAX_CHARS = 45_000  # conservative limit for the prompt

    pages_repr = ""
    truncated_at: Optional[int] = None
    for p in pages:
        chunk = f"\n=== Página {p['page_num']} ===\n{p['text'][:900]}\n"
        if len(pages_repr) + len(chunk) > MAX_CHARS:
            truncated_at = p["page_num"]
            break
        pages_repr += chunk

    total_pages = pages[-1]["page_num"] if pages else 1
    truncation_note = (
        f"\n[Nota: el documento tiene {total_pages} páginas. "
        f"Se muestra hasta la página {truncated_at - 1} para el análisis. "
        f"Extrapolá los nodos para cubrir hasta la página {total_pages}.]"
        if truncated_at
        else ""
    )

    prompt = f"""Analizá este documento y generá una estructura jerárquica tipo tabla de contenidos (tree structure).
{truncation_note}

Documento (extracto por página):
{pages_repr}

Generá un JSON array de nodos. Cada nodo DEBE tener exactamente estos campos:
- "node_id": string único (ej: "001", "002", "001-1", "001-2")
- "title": string — título descriptivo de la sección
- "summary": string — resumen de 1-2 oraciones del contenido
- "start_page": integer — página de inicio
- "end_page": integer — página de fin (inclusive)
- "level": integer — nivel jerárquico (1=principal, 2=subsección, 3=sub-subsección)
- "parent_id": string o null — node_id del padre; null para nodos de nivel 1

Reglas:
- Toda página del documento debe estar cubierta por al menos un nodo
- Los nodos de nivel 1 cubren rangos amplios; los de nivel 2-3 dividen ese rango
- Los títulos deben ser específicos y descriptivos (no "Sección 1")
- Los resúmenes deben capturar la esencia del contenido para poder razonar sin leer el texto

Retorná SOLO el JSON array, sin texto adicional ni code fences."""

    response = llm.call(prompt)
    nodes = _extract_json(response)

    # Basic validation
    for node in nodes:
        node.setdefault("level", 1)
        node.setdefault("parent_id", None)

    return nodes


# ── Step 2: Reasoning-based retrieval ─────────────────────────────────────────

def retrieve_nodes(query: str, tree_nodes: List[Dict], llm: LLMClient) -> Dict:
    """
    Ask the LLM to reason over the tree (titles + summaries only, NO full text)
    and identify which nodes are most likely to contain the answer.

    This is the core insight of vectorless RAG: the LLM uses reasoning
    instead of vector similarity to determine relevance.
    """
    tree_slim = [
        {
            "node_id": n["node_id"],
            "title": n["title"],
            "summary": n["summary"],
            "pages": f"{n['start_page']}-{n['end_page']}",
            "level": n.get("level", 1),
        }
        for n in tree_nodes
    ]

    prompt = f"""Sos un experto analizando un documento para encontrar las secciones relevantes a una pregunta.

Pregunta: {query}

Árbol del documento (solo títulos y resúmenes — sin texto completo):
{json.dumps(tree_slim, indent=2, ensure_ascii=False)}

Tarea:
1. Razoná paso a paso qué secciones probablemente contienen la respuesta
2. Seleccioná los node_ids de las secciones más relevantes (máximo 5)

Importante: elegí solo los nodos que realmente aporten información — menos es más.

Respondé SOLO con este JSON (sin code fences):
{{
    "thinking": "Tu razonamiento paso a paso, explicando por qué cada sección elegida es relevante",
    "node_ids": ["node_id_1", "node_id_2"]
}}"""

    response = llm.call(prompt)
    return _extract_json(response)


# ── Step 3: Context extraction ────────────────────────────────────────────────

def get_context_from_nodes(
    node_ids: List[str],
    tree_nodes: List[Dict],
    pages: List[Dict],
) -> Tuple[str, List[Dict]]:
    """Extract full text from the pages covered by each selected node."""
    node_map = {n["node_id"]: n for n in tree_nodes}
    page_map = {p["page_num"]: p["text"] for p in pages}

    context_parts = []
    sources = []

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            continue
        sources.append(node)
        page_texts = []
        for pg in range(node["start_page"], node["end_page"] + 1):
            text = page_map.get(pg, "")
            if text:
                page_texts.append(f"[Página {pg}]\n{text}")
        section_text = f"### {node['title']}\n" + "\n\n".join(page_texts)
        context_parts.append(section_text)

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


# ── Step 4: Answer generation ─────────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    sources: List[Dict],
    llm: LLMClient,
) -> str:
    """Generate the final answer using only the retrieved context."""
    sources_str = "\n".join(
        f"  - '{s['title']}' (páginas {s['start_page']}-{s['end_page']})"
        for s in sources
    )

    prompt = f"""Respondé la siguiente pregunta basándote ÚNICAMENTE en el contexto provisto.
Sé preciso y referenciá páginas específicas cuando sea relevante.

Pregunta: {query}

Contexto del documento:
{context}

Fuentes consultadas:
{sources_str}

Si el contexto no contiene suficiente información para responder, indicalo explícitamente.
Respondé en el mismo idioma que la pregunta."""

    return llm.call(prompt)
