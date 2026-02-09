import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from nlp_preprocessing import extract_entities, extract_triples

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Mini Knowledge Graph Builder", layout="wide")

st.title("🧠 Mini Knowledge Graph Builder")
st.caption("Generate a knowledge graph from plain text using NLTK")

# -------------------------------
# USER INPUT
# -------------------------------
text = st.text_area(
    "✍️ Enter a paragraph:",
    height=160,
    placeholder="Google was founded by Shahista. Shahista is CEO of Google."
)

if st.button("🚀 Generate Knowledge Graph"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # -------------------------------
    # NLP
    # -------------------------------
    entities = extract_entities(text)
    triples = extract_triples(text)

    # -------------------------------
    # DISPLAY ENTITIES
    # -------------------------------
    st.subheader("📌 Extracted Entities")
    if entities:
        for e, t in entities:
            st.write(f"**{e}** — {t}")
    else:
        st.info("No entities found.")

    # -------------------------------
    # DISPLAY TRIPLES
    # -------------------------------
    st.subheader("🔗 Extracted Triples")

    if not triples:
        st.warning("No meaningful relations found.")
        st.stop()

    for s, r, o in triples:
        st.write(f"**{s} → {r} → {o}**")

    # -------------------------------
    # GRAPH BUILDING
    # -------------------------------
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_edge(s, o, label=r)

    # -------------------------------
    # GRAPH VISUALIZATION
    # -------------------------------
    st.subheader("📊 Knowledge Graph")

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2800,
        node_color="#90CAF9",
        font_size=10,
        font_weight="bold",
        edge_color="#555"
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    st.pyplot(plt)