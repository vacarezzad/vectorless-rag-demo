import streamlit as st
import pandas as pd


def render():
    st.header("¿Qué es Vectorless RAG?")

    col_intro, col_stat = st.columns([3, 2])

    with col_intro:
        st.markdown("""
Vectorless RAG es un enfoque alternativo al RAG tradicional que **elimina completamente
la necesidad de vectores, embeddings y bases de datos vectoriales**.

En lugar de convertir el texto en representaciones numéricas para hacer búsquedas por similitud,
el sistema:

1. **Construye un árbol jerárquico** del documento — similar a una tabla de contenidos enriquecida
2. **Usa el LLM para razonar** sobre ese árbol y encontrar las secciones relevantes a la pregunta
3. **Extrae el texto** exacto de esas secciones y genera la respuesta

> *Inspirado en cómo un experto humano navega un documento: primero revisa el índice,
> identifica las secciones relevantes, las lee en detalle y formula su respuesta.*
        """)

    with col_stat:
        st.success("""
**Resultado empírico**

PageIndex (la implementación de referencia de Vectorless RAG)
logró **98.7% de accuracy en FinanceBench** — el benchmark
estándar de análisis de documentos financieros — superando
ampliamente a todas las soluciones de RAG vectorial.
        """)

    st.divider()

    # ── Architecture diagrams ────────────────────────────────────────────────
    st.subheader("Arquitectura: RAG Tradicional vs Vectorless RAG")

    col_trad, col_vless = st.columns(2)

    with col_trad:
        st.markdown("#### ❌ RAG Tradicional")
        st.graphviz_chart("""
digraph traditional {
    rankdir=TB
    node [shape=box style=filled fontname="Arial" fontsize=11 width=2.2]
    edge [fontsize=10]

    A  [label="Documento PDF"              fillcolor="#FFF3CD" color="#856404"]
    B  [label="Chunking artificial\\n(fragmentación por tokens)"
                                           fillcolor="#F8D7DA" color="#721C24"]
    C  [label="Embedding Model\\n(texto → vector numérico)"
                                           fillcolor="#F8D7DA" color="#721C24"]
    D  [label="Vector Database\\n(Pinecone / Chroma / Weaviate)"
                                           fillcolor="#F8D7DA" color="#721C24"]
    E  [label="Query del usuario"          fillcolor="#E2E3E5" color="#383D41"]
    F  [label="Embed de la query"          fillcolor="#F8D7DA" color="#721C24"]
    G  [label="Similarity Search\\n(cosine / dot product)"
                                           fillcolor="#F8D7DA" color="#721C24"]
    H  [label="Chunks recuperados\\n(posiblemente sin contexto)"
                                           fillcolor="#FFE5D0" color="#6C3D14"]
    I  [label="LLM → Respuesta"            fillcolor="#D4EDDA" color="#155724"]

    A -> B -> C -> D
    E -> F -> G
    D -> G
    G -> H -> I
}
        """)

    with col_vless:
        st.markdown("#### ✅ Vectorless RAG")
        st.graphviz_chart("""
digraph vectorless {
    rankdir=TB
    node [shape=box style=filled fontname="Arial" fontsize=11 width=2.2]
    edge [fontsize=10]

    A  [label="Documento PDF"              fillcolor="#FFF3CD" color="#856404"]
    B  [label="Tree Generation\\n(LLM genera árbol jerárquico)"
                                           fillcolor="#CCE5FF" color="#004085"]
    C  [label="Árbol de secciones\\n(JSON con títulos y resúmenes)"
                                           fillcolor="#CCE5FF" color="#004085"]
    D  [label="Query del usuario"          fillcolor="#E2E3E5" color="#383D41"]
    E  [label="LLM Reasoning\\n(razona sobre el árbol)"
                                           fillcolor="#CCE5FF" color="#004085"]
    F  [label="Nodos relevantes\\n(+ razonamiento explícito)"
                                           fillcolor="#B8DAFF" color="#004085"]
    G  [label="Extracción de texto\\n(solo secciones elegidas)"
                                           fillcolor="#CCE5FF" color="#004085"]
    H  [label="LLM → Respuesta"            fillcolor="#D4EDDA" color="#155724"]

    A -> B -> C
    D -> E
    C -> E
    E -> F -> G -> H
}
        """)

    st.divider()

    # ── Problems ─────────────────────────────────────────────────────────────
    st.subheader("⚠️ Problemas del RAG Tradicional")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("""
**Chunking artificial**

Fragmentar el texto en pedazos de tamaño fijo rompe el contexto
natural del documento. Una respuesta que abarca varias páginas
puede quedar partida en chunks que nunca se recuperan juntos.
        """)
    with c2:
        st.error("""
**Similitud ≠ Relevancia**

La búsqueda vectorial ("vibe retrieval") encuentra texto
*semánticamente parecido* a la query, no necesariamente el texto
que *lógicamente* contiene la respuesta.
        """)
    with c3:
        st.error("""
**Opacidad total**

No hay forma de saber por qué se recuperaron ciertos chunks.
La búsqueda vectorial es una caja negra — difícil de debuggear,
auditar o explicar ante un usuario final.
        """)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.error("""
**Infraestructura compleja**

Requiere: modelo de embeddings + vector database + pipeline de
indexación. Alto costo operativo, más dependencias, más superficie
de fallo.
        """)
    with c5:
        st.error("""
**Degradación en docs estructurados**

En contratos, reportes financieros y manuales técnicos — donde la
jerarquía y estructura importan — el RAG vectorial pierde
información contextual clave al fragmentar.
        """)
    with c6:
        st.error("""
**Re-indexación costosa**

Cualquier actualización del documento requiere re-generar todos
los embeddings y actualizar la base de datos vectorial.
        """)

    st.divider()

    # ── Comparison table ──────────────────────────────────────────────────────
    st.subheader("📊 Tabla Comparativa")

    data = {
        "Característica": [
            "Vector Database",
            "Modelo de Embeddings",
            "Chunking del texto",
            "Mecanismo de búsqueda",
            "Trazabilidad del resultado",
            "Complejidad de infraestructura",
            "Documentos estructurados",
            "Explicabilidad",
            "Re-indexación ante cambios",
            "Accuracy en FinanceBench",
        ],
        "RAG Tradicional": [
            "✅ Requerido",
            "✅ Requerido",
            "✅ Requerido (artificial)",
            "Similitud semántica (cosine/dot)",
            "❌ Opaca",
            "Alta (3+ componentes extra)",
            "⚠️ Pierde jerarquía",
            "❌ Caja negra",
            "Costosa (re-embed todo)",
            "~70–80%",
        ],
        "Vectorless RAG": [
            "❌ No necesario",
            "❌ No necesario",
            "❌ No necesario",
            "Razonamiento LLM sobre árbol",
            "✅ Explícita y auditable",
            "Baja (solo el LLM)",
            "✅ Preserva estructura natural",
            "✅ Razonamiento step-by-step",
            "Ligera (solo tree gen)",
            "**98.7%** (PageIndex / Mafin 2.5)",
        ],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
