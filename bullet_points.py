import streamlit as st

def _extract_text(result):
    return getattr(result, "content", getattr(result, "text", str(result)))

def bullet_points_ui(llm, retriever):
    with st.expander("📋 Break Document into Bullet Points"):
        if st.button("Generate Bullet Points"):
            with st.spinner("Generating bullet points…"):
                chunks = st.session_state.get("chunks", [])
                if not chunks:
                    docs = retriever.vectorstore.similarity_search("outline", k=20)
                    chunks = docs

                # Summarize each chunk into bullet points, then merge
                partial_bullets = []
                for d in chunks[:15]:  # limit to first 15 chunks to control token use
                    part_prompt = (
                        "From this excerpt, extract structured bullet points covering all key ideas.\n\n"
                        f"Excerpt:\n{d.page_content}\n\nBullet Points (bulleted list):"
                    )
                    partial_bullets.append(_extract_text(llm.invoke(part_prompt)))

                combined = "\n".join(partial_bullets)
                merge_prompt = (
                    "Merge and deduplicate the bullet points below into a coherent, well-organized outline. "
                    "Group related bullets under appropriate subheadings if helpful. Use only the provided content.\n\n"
                    f"Bullets to merge:\n{combined}\n\nFinal Bullet Outline:"
                )
                bullet_text = _extract_text(llm.invoke(merge_prompt))
                st.markdown(bullet_text)