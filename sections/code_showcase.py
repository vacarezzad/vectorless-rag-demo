import streamlit as st


def render():
    st.header("💻 Implementación: Vectorless RAG desde cero")
    st.markdown(
        "Esta es la implementación **exacta** que corre en la pestaña Demo. "
        "Sin librerías de RAG, sin vector DB — solo Python y el LLM de tu elección."
    )

    # ── Step 1 ────────────────────────────────────────────────────────────────
    st.subheader("Paso 1 — Generar el árbol jerárquico del documento")
    st.markdown(
        "El LLM lee un extracto de cada página y produce una estructura tipo TOC "
        "con títulos, resúmenes y rangos de páginas. **Esto reemplaza el chunking y el embedding.**"
    )
    st.code(
        '''
def generate_tree(pages: list[dict], llm: LLMClient) -> list[dict]:
    """
    Convierte las páginas del PDF en un árbol tipo tabla de contenidos.
    El LLM lee el documento y genera una estructura jerárquica con:
      - Título de cada sección
      - Resumen de 1-2 oraciones
      - Rango de páginas
      - Nivel jerárquico y nodo padre
    """
    pages_repr = "\\n".join(
        f"=== Página {p[\'page_num\']} ===\\n{p[\'text\'][:900]}"
        for p in pages
    )

    prompt = f"""Analizá este documento y generá un árbol jerárquico tipo TOC.

Documento:
{pages_repr}

Generá un JSON array donde cada nodo tiene:
  node_id, title, summary, start_page, end_page, level, parent_id

Retorná SOLO el JSON array."""

    response = llm.call(prompt)
    return json.loads(response)


# Ejemplo de árbol generado:
# [
#   {
#     "node_id": "001",
#     "title": "Introducción y Contexto",
#     "summary": "Presenta los objetivos y el marco general del documento.",
#     "start_page": 1, "end_page": 3,
#     "level": 1, "parent_id": null
#   },
#   {
#     "node_id": "001-1",
#     "title": "Antecedentes del Proyecto",
#     "summary": "Historia, motivación y decisiones de diseño iniciales.",
#     "start_page": 1, "end_page": 2,
#     "level": 2, "parent_id": "001"
#   },
#   ...
# ]
''',
        language="python",
    )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    st.subheader("Paso 2 — Recuperación basada en razonamiento (tree search)")
    st.markdown(
        "El LLM recibe el árbol **sin el texto completo** (solo títulos + resúmenes) "
        "y razona cuáles nodos son relevantes a la query. "
        "El resultado incluye el razonamiento paso a paso — completamente trazable. "
        "**Prompt fiel al repositorio oficial [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex).**"
    )
    st.code(
        '''
def retrieve_nodes(query: str, tree_nodes: list[dict], llm: LLMClient) -> dict:
    """
    El LLM examina el árbol (solo node_id + title + summary, SIN texto completo)
    y devuelve los nodos relevantes con razonamiento explícito.

    Fuente: prompt oficial de VectifyAI/PageIndex.
    """
    # Eliminamos el texto — solo enviamos estructura al LLM
    tree_without_text = [
        {"node_id": n["node_id"], "title": n["title"], "summary": n["summary"]}
        for n in tree_nodes
    ]

    # Prompt oficial de PageIndex
    prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}
Document tree structure: {json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{"thinking": "<your step-by-step reasoning>", "node_list": ["node_id_1", "node_id_2"]}}"""

    response = llm.call(prompt)
    return json.loads(response)


# Ejemplo de resultado:
# {
#   "thinking": "The question is about regulatory compliance.
#                Node '004' covers exactly that topic per its summary.
#                Node '004-2' dives into the specific articles.
#                Other sections are unrelated.",
#   "node_list": ["004", "004-2"]     # <-- campo oficial: node_list
# }
''',
        language="python",
    )

    # ── Step 3 ────────────────────────────────────────────────────────────────
    st.subheader("Paso 3 — Extracción de contexto y generación de respuesta")
    st.markdown(
        "Se extrae el texto **solo** de las páginas cubiertas por los nodos elegidos. "
        "Contexto mínimo y preciso, sin tokens desperdiciados en chunks irrelevantes."
    )
    st.code(
        '''
def get_context_from_nodes(
    node_ids: list[str],
    tree_nodes: list[dict],
    pages: list[dict],
) -> tuple[str, list[dict]]:
    """Extrae el texto de las páginas cubiertas por cada nodo seleccionado."""
    node_map = {n["node_id"]: n for n in tree_nodes}
    page_map = {p["page_num"]: p["text"] for p in pages}

    context_parts, sources = [], []
    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            continue
        sources.append(node)
        page_texts = [
            f"[Página {pg}]\\n{page_map[pg]}"
            for pg in range(node["start_page"], node["end_page"] + 1)
            if page_map.get(pg)
        ]
        context_parts.append(f"### {node[\'title\']}\\n" + "\\n\\n".join(page_texts))

    return "\\n\\n---\\n\\n".join(context_parts), sources


def generate_answer(query, context, sources, llm):
    sources_str = "\\n".join(
        f"  - \'{s[\'title\']}\' (páginas {s[\'start_page\']}-{s[\'end_page\']})"
        for s in sources
    )
    prompt = f"""Respondé basándote ÚNICAMENTE en el contexto provisto.

Pregunta: {query}
Contexto: {context}
Fuentes: {sources_str}"""

    return llm.call(prompt)
''',
        language="python",
    )

    # ── Full pipeline ─────────────────────────────────────────────────────────
    st.subheader("Pipeline completo")
    st.code(
        '''
def vectorless_rag(pdf_path: str, query: str, llm: LLMClient) -> dict:
    pages    = extract_pages(pdf_path)          # 1. PDF → páginas

    tree     = generate_tree(pages, llm)        # 2. Páginas → árbol (una vez)

    retrieval = retrieve_nodes(query, tree, llm) # 3. Query → nodos relevantes

    context, sources = get_context_from_nodes(  # 4. Nodos → texto
        retrieval["node_list"], tree, pages      # campo oficial: node_list
    )
    answer = generate_answer(query, context, sources, llm)  # 5. → respuesta

    return {
        "answer":   answer,
        "thinking": retrieval["thinking"],      # razonamiento auditable
        "sources":  sources,                    # secciones referenciadas
    }
''',
        language="python",
    )

    # ── LLM abstraction ───────────────────────────────────────────────────────
    st.subheader("Abstracción multi-LLM")
    st.markdown(
        "Una interfaz unificada que soporta OpenAI, Anthropic, Gemini y Mistral. "
        "El resto del sistema no sabe qué LLM está usando."
    )
    st.code(
        '''
class LLMClient:
    def call(self, prompt: str, system: str | None = None) -> str:
        handlers = {
            "OpenAI":             self._call_openai,
            "Anthropic (Claude)": self._call_anthropic,
            "Google (Gemini)":    self._call_gemini,
            "Mistral":            self._call_mistral,
        }
        return handlers[self.provider](prompt, system)

    def _call_openai(self, prompt, system):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        ).choices[0].message.content.strip()

    def _call_anthropic(self, prompt, system):
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        return client.messages.create(
            model=self.model, max_tokens=4096,
            system=system or "",
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text.strip()

    def _call_gemini(self, prompt, system):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        full_prompt = f"{system}\\n\\n{prompt}" if system else prompt
        return genai.GenerativeModel(self.model).generate_content(full_prompt).text.strip()

    def _call_mistral(self, prompt, system):
        from mistralai import Mistral
        client = Mistral(api_key=self.api_key)
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return client.chat.complete(
            model=self.model, messages=messages, temperature=0
        ).choices[0].message.content.strip()
''',
        language="python",
    )
