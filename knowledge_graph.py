import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from io import StringIO

def _extract_text(result):
    return getattr(result, "content", getattr(result, "text", str(result)))

def _parse_triples(text: str):
    triples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Expected format: subject | relation | object
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples

def _build_graph(triples):
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, label=r)
    return G

def _render_graph(G: nx.DiGraph) -> str:
    net = Network(height="550px", width="100%", bgcolor="#0B1020")
    net.barnes_hut()
    for node in G.nodes():
        net.add_node(node, label=node)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("label", ""))
    html_path = "kg.html"
    # Avoid PyVis's notebook=True path which can break in non-notebook contexts
    net.write_html(html_path, open_browser=False)
    # Read back the generated HTML to embed
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

def knowledge_graph_ui(llm, retriever):
    with st.expander("🕸️ Knowledge Graph (experimental)", expanded=False):
        st.caption("Extract entities and relations from your document and view them as a graph.")
        sample_k = st.slider("Sample chunks", 5, 30, 12)
        if st.button("Build Graph"):
            with st.spinner("Extracting triples and building graph…"):
                # Sample some representative chunks
                try:
                    docs = retriever.get_relevant_documents("key entities and relationships")[:sample_k]
                except Exception:
                    docs = []
                basis = "\n\n".join(d.page_content for d in docs)
                prompt = (
                    "Extract key factual triples from the following document excerpts in the format: subject | relation | object. "
                    "Keep each triple on its own line and avoid duplicates. Focus on concrete facts and definitions.\n\n"
                    f"Excerpts:\n{basis}\n\nTriples:"
                )
                triples_text = _extract_text(llm.invoke(prompt))
                triples = _parse_triples(triples_text)
                if not triples:
                    st.warning("No triples detected. Try increasing sample size or using a different document.")
                    return
                G = _build_graph(triples)
                html = _render_graph(G)
                components.html(html, height=560, scrolling=False)
