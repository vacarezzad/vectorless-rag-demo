# 🌲 Vectorless RAG — Demo

Demo técnica interactiva de **Vectorless RAG**: una alternativa al RAG tradicional que elimina completamente la necesidad de vectores, embeddings y bases de datos vectoriales.

Construida con [Streamlit](https://streamlit.io) y compatible con **OpenAI, Anthropic (Claude), Google (Gemini) y Mistral**.

---

## ¿Qué es Vectorless RAG?

En lugar de convertir texto en vectores numéricos para hacer búsquedas por similitud, Vectorless RAG:

1. **Genera un árbol jerárquico** del documento (similar a una tabla de contenidos enriquecida)
2. **Usa el LLM para razonar** sobre ese árbol y encontrar las secciones relevantes a la pregunta
3. **Extrae el texto exacto** de esas secciones y genera la respuesta

> Inspirado en cómo un experto humano navega un documento: revisa el índice, identifica las secciones relevantes, las lee en detalle y formula su respuesta.

La implementación de referencia es [PageIndex de VectifyAI](https://github.com/VectifyAI/PageIndex), que logró **98.7% de accuracy en FinanceBench**, superando a todas las soluciones de RAG vectorial.

---

## Estructura del proyecto

```
vectorless-rag-demo/
│
├── app.py                  # Entry point — configuración de página y tabs
│
├── core/                   # Lógica de negocio (sin dependencias de UI)
│   ├── llm_client.py       # Abstracción multi-LLM (OpenAI / Claude / Gemini / Mistral)
│   ├── pdf_utils.py        # Extracción de texto por página desde PDF
│   └── vectorless_rag.py   # Pipeline completo: tree gen → retrieval → answer
│
├── sections/               # Secciones de la UI (una por tab)
│   ├── educational.py      # Tab "¿Qué es?" — explicación, diagramas, tabla comparativa
│   ├── code_showcase.py    # Tab "Implementación" — código comentado paso a paso
│   └── demo.py             # Tab "Demo" — upload PDF, árbol, chat
│
└── requirements.txt        # Dependencias del proyecto
```

---

## Pipeline de Vectorless RAG

```
PDF
 │
 ▼
[Paso 1] generate_tree()
  └─ El LLM lee un extracto de cada página y genera un árbol JSON:
     { node_id, title, summary, start_page, end_page, level, parent_id }
 │
 ▼
[Paso 2] retrieve_nodes()   ← prompt oficial de PageIndex
  └─ El LLM recibe SOLO títulos + resúmenes (sin texto completo)
     y razona qué nodos son relevantes a la query.
     Devuelve: { thinking, node_list }
 │
 ▼
[Paso 3] get_context_from_nodes()
  └─ Extrae el texto completo de las páginas cubiertas por los nodos elegidos.
 │
 ▼
[Paso 4] generate_answer()
  └─ Genera la respuesta final usando solo el contexto recuperado.
```

### Prompt de retrieval (fiel al repositorio oficial de PageIndex)

```python
prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}
Document tree structure: {json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{"thinking": "<your step-by-step reasoning>", "node_list": ["node_id_1", "node_id_2"]}}"""
```

---

## RAG Tradicional vs Vectorless RAG

| Característica | RAG Tradicional | Vectorless RAG |
|---|---|---|
| Vector Database | ✅ Requerido | ❌ No necesario |
| Modelo de Embeddings | ✅ Requerido | ❌ No necesario |
| Chunking del texto | ✅ Requerido (artificial) | ❌ No necesario |
| Mecanismo de búsqueda | Similitud semántica (cosine) | Razonamiento LLM |
| Trazabilidad | ❌ Opaca | ✅ Explícita y auditable |
| Infraestructura | Alta (3+ componentes) | Baja (solo el LLM) |
| Docs estructurados | ⚠️ Pierde jerarquía | ✅ Preserva estructura |
| Explicabilidad | ❌ Caja negra | ✅ Step-by-step reasoning |
| Accuracy en FinanceBench | ~70–80% | **98.7%** |

---

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/vacarezzad/vectorless-rag-demo.git
cd vectorless-rag-demo
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Las librerías de cada proveedor LLM están incluidas en `requirements.txt`. Podés instalar solo las que vayas a usar:
> ```bash
> pip install openai          # OpenAI
> pip install anthropic       # Anthropic (Claude)
> pip install google-generativeai  # Google Gemini
> pip install mistralai       # Mistral
> ```

### 3. Lanzar la app

```bash
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

---

## Uso de la app

### Tab 1 — ¿Qué es?
Explicación conceptual de Vectorless RAG, diagramas de arquitectura (RAG tradicional vs Vectorless), listado de problemas del RAG clásico y tabla comparativa.

### Tab 2 — Implementación
Código Python comentado de cada paso del pipeline: generación del árbol, retrieval por razonamiento, extracción de contexto y generación de respuesta. Incluye la abstracción multi-LLM.

### Tab 3 — Demo
1. Configurá tu LLM en el panel lateral (proveedor, modelo, API key)
2. Subí cualquier PDF extenso
3. Hacé click en **Generar árbol jerárquico** — el LLM construye el árbol del documento
4. Visualizá el árbol (lista jerárquica o diagrama)
5. Chateá con el documento — cada respuesta muestra:
   - 🧠 El razonamiento del LLM (qué secciones eligió y por qué)
   - 📄 Las fuentes consultadas (título + rango de páginas)
   - 🔍 Los `node_ids` seleccionados

---

## Proveedores LLM soportados

| Proveedor | Modelos disponibles |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| Anthropic (Claude) | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| Google (Gemini) | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` |
| Mistral | `mistral-large-latest`, `mistral-small-latest`, `open-mixtral-8x7b` |

---

## Referencias

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — implementación de referencia
- [What is PageIndex? How to build a Vectorless RAG system](https://medium.com/@visrow/what-is-pageindex-how-to-build-a-vectorless-rag-system-no-embeddings-no-vector-db-dc097fae3071)
- [Vectorless Reasoning-Based RAG — Microsoft Tech Community](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/vectorless-reasoning-based-rag-a-new-approach-to-retrieval-augmented-generation/4502238)
- [Colab notebook oficial de PageIndex](https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb)
