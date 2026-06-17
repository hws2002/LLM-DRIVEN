"""Prompt templates for RAG."""


class PromptFactory:
    @staticmethod
    def rag(context: str, query: str) -> str:
        return (
            "Use the following context to answer the question. "
            "Please be specific as you can. "
            "If the context is not enough, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

    @staticmethod
    def rag_with_profile(context: str, query: str, profile: str) -> str:
        """RAG prompt that fuses the document context with the user's
        Macro-Graph knowledge profile, so the answer fits the user's
        personal context (interests, learning style, recurring patterns)."""
        if not profile:
            return PromptFactory.rag(context, query)
        return (
            "Use the following document context to answer the question. "
            "Be as specific as you can. "
            "If the context is not enough, say you don't know.\n"
            "Tailor the answer to the user's knowledge profile below: relate the "
            "concept to the user's existing interests and recurring patterns when "
            "relevant, but never invent facts that are not in the context.\n\n"
            f"{profile}\n\n"
            f"Document context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

    @staticmethod
    def summary(context: str, topic: str) -> str:
        return (
            f"Summarize the following context about '{topic}'. "
            "Focus on the key concepts and relationships.\n\n"
            f"Context:\n{context}\n\n"
            f"Summary:"
        )

    @staticmethod
    def related_questions(entities: list[str], query: str = "") -> str:
        entity_list = ", ".join(entities) if entities else "(none)"
        query = (query or "").strip()
        if query:
            return (
                "Generate 5 follow-up questions related to the user query and entities below. "
                "Questions should be specific and non-duplicative.\n\n"
                f"User query: {query}\n"
                f"Entities: {entity_list}\n\n"
                "Questions:"
            )
        return (
            "Generate 5 related conversation questions that could be asked based on "
            f"these entities: {entity_list}\n\n"
            "Questions:"
        )

    @staticmethod
    def no_context(query: str) -> str:
        return (
            f"Answer the following question based on your general knowledge:\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
