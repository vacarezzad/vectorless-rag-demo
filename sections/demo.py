import streamlit as st

from core.llm_client import LLMClient
from core.pdf_utils import extract_pages
from core.vectorless_rag import (
    generate_tree,
    retrieve_nodes,
    get_context_from_nodes,
    generate_answer,
)


def _render_tree(tree: list) -> None:
    """Render the tree as a nested visual list."""
    # Build parent → children map
    children: dict[str | None, list] = {}
    for node in tree:
        pid = node.get("parent_id")
        children.setdefault(pid, []).append(node)

    def _render_node(node: dict, depth: int) -> None:
        indent = "&nbsp;" * depth * 6
        level_colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd"}
        color = level_colors.get(node.get("level", 1), "#666")
        st.markdown(
            f"{indent}"
            f"<span style='color:{color}; font-weight:bold'>[{node['node_id']}]</span> "
            f"**{node['title']}** "
            f"<span style='color:#888; font-size:0.85em'>págs. {node['start_page']}–{node['end_page']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{indent}&nbsp;&nbsp;<span style='color:#555; font-size:0.85em'>{node['summary']}</span>",
            unsafe_allow_html=True,
        )
        for child in children.get(node["node_id"], []):
            _render_node(child, depth + 1)

    for root in children.get(None, []):
        _render_node(root, 0)
        st.markdown("---")


def _render_tree_diagram(tree: list) -> None:
    """Render the tree as a Graphviz diagram (up to 20 nodes for readability)."""
    nodes_to_show = tree[:20]

    dot_nodes = []
    dot_edges = []

    level_colors = {1: "#CCE5FF", 2: "#D4EDDA", 3: "#FFF3CD"}

    for node in nodes_to_show:
        nid = node["node_id"].replace("-", "_")
        color = level_colors.get(node.get("level", 1), "#E2E3E5")
        label = f"{node['title']}\\npágs. {node['start_page']}–{node['end_page']}"
        label = label.replace('"', "'")
        dot_nodes.append(f'  n{nid} [label="{label}" fillcolor="{color}"]')
        if node.get("parent_id"):
            pid = node["parent_id"].replace("-", "_")
            dot_edges.append(f"  n{pid} -> n{nid}")

    dot = (
        "digraph tree {\n"
        "  rankdir=TB\n"
        '  node [shape=box style=filled fontname="Arial" fontsize=10]\n'
        + "\n".join(dot_nodes)
        + "\n"
        + "\n".join(dot_edges)
        + "\n}"
    )
    st.graphviz_chart(dot)


def render() -> None:
    st.header("🔬 Demo Práctica")
    st.markdown(
        "Subí cualquier PDF extenso, generá el árbol y chateá con el documento. "
        "Cada respuesta muestra el razonamiento del LLM y las secciones consultadas."
    )

    # ── Sidebar: LLM config ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración del LLM")

        provider = st.selectbox(
            "Proveedor",
            options=list(LLMClient.PROVIDERS.keys()),
            format_func=lambda x: f"{LLMClient.PROVIDERS[x]['icon']} {x}",
        )
        model = st.selectbox("Modelo", options=LLMClient.PROVIDERS[provider]["models"])
        api_key = st.text_input(
            f"API Key — {provider}",
            type="password",
            placeholder="sk-...",
        )

        if api_key:
            st.success("API Key configurada ✓")
        else:
            st.warning("Ingresá tu API Key para usar el demo")

        st.divider()
        st.caption("🌲 Vectorless RAG — Demo técnica")

    if not api_key:
        st.info("👈 Configurá tu LLM en el panel lateral para comenzar.")
        return

    llm = LLMClient(provider=provider, api_key=api_key, model=model)

    # ── Step 1: Upload PDF ────────────────────────────────────────────────────
    st.subheader("1️⃣  Subir documento PDF")
    uploaded = st.file_uploader("Elegí un PDF extenso", type=["pdf"])

    if not uploaded:
        st.info("Subí un PDF para continuar.")
        return

    # ── Step 2: Tree generation ───────────────────────────────────────────────
    st.subheader("2️⃣  Generar árbol del documento")

    cache_key = f"tree__{uploaded.name}__{uploaded.size}"

    if cache_key not in st.session_state:
        if st.button("🌲 Generar árbol jerárquico", type="primary"):
            with st.spinner("Extrayendo páginas del PDF…"):
                pages = extract_pages(uploaded)
                st.session_state["pages"] = pages

            with st.spinner(
                f"Generando árbol con **{model}**… (puede tardar ~30 s según el documento)"
            ):
                try:
                    tree = generate_tree(pages, llm)
                    st.session_state[cache_key] = tree
                    st.session_state["chat_history"] = []
                    st.success(
                        f"✅ Árbol generado — {len(tree)} nodos | {len(pages)} páginas"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error al generar el árbol: {exc}")
        return

    tree: list = st.session_state[cache_key]
    pages: list = st.session_state.get("pages", [])

    # ── Tree visualization ────────────────────────────────────────────────────
    st.success(f"✅ Árbol listo — {len(tree)} nodos | {len(pages)} páginas")

    view_mode = st.radio(
        "Ver árbol como:",
        ["Lista jerárquica", "Diagrama"],
        horizontal=True,
    )

    with st.expander("🌲 Ver árbol del documento", expanded=False):
        if view_mode == "Lista jerárquica":
            _render_tree(tree)
        else:
            if len(tree) > 20:
                st.caption(
                    f"Mostrando los primeros 20 nodos de {len(tree)} para legibilidad."
                )
            _render_tree_diagram(tree)

    # ── Step 3: Chat ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("3️⃣  Chat con el documento")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("thinking"):
                    with st.expander("🧠 Razonamiento del LLM — qué secciones eligió y por qué"):
                        st.info(msg["thinking"])
                if msg.get("sources"):
                    with st.expander("📄 Fuentes consultadas"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"- **{s['title']}** — páginas {s['start_page']}–{s['end_page']}"
                            )
                if msg.get("node_list"):
                    with st.expander("🔍 Nodos seleccionados"):
                        st.code(str(msg["node_list"]))

    # New message
    query = st.chat_input("Hacé una pregunta sobre el documento…")

    if query:
        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Razonando sobre el árbol del documento…"):
                    retrieval = retrieve_nodes(query, tree, llm)

                with st.spinner("Extrayendo contexto y generando respuesta…"):
                    node_list = retrieval.get("node_list", [])
                    context, sources = get_context_from_nodes(node_list, tree, pages)
                    answer = generate_answer(query, context, sources, llm)

                st.markdown(answer)

                with st.expander("🧠 Razonamiento del LLM — qué secciones eligió y por qué"):
                    st.info(retrieval.get("thinking", "Sin razonamiento disponible"))

                with st.expander("📄 Fuentes consultadas"):
                    for s in sources:
                        st.markdown(
                            f"- **{s['title']}** — páginas {s['start_page']}–{s['end_page']}"
                        )

                with st.expander("🔍 Nodos seleccionados (node_list)"):
                    st.code(str(node_list))

                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "thinking": retrieval.get("thinking", ""),
                        "sources": sources,
                        "node_list": node_list,
                    }
                )

            except Exception as exc:
                st.error(f"Error al procesar la pregunta: {exc}")

    # Reset button
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Limpiar conversación"):
            st.session_state["chat_history"] = []
            st.rerun()
