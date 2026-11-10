"""Wiring: build the real components (store, retriever, chain, graph) from Settings.

The CLI and the Streamlit UI both start here, so they always use the same pieces. Tests skip this
file and build the components themselves with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings


@dataclass
class Runtime:
    settings: Settings
    store: object

    @classmethod
    def open(cls, settings: Settings | None = None, embedding=None) -> "Runtime":
        from .store import open_store

        settings = settings or get_settings()
        return cls(settings=settings, store=open_store(settings, embedding))

    def topics(self) -> list[str]:
        """File names and headings: the router's only hint at what the documents are about."""
        from .store import topics

        return topics(self.store)

    def retriever(self, k: int | None = None, mmr: bool = False):
        from .store import build_retriever

        return build_retriever(self.store, k or self.settings.top_k, mmr=mmr)

    def rag_chain(self, k: int | None = None, mmr: bool = False, models=None):
        from .agent import build_models
        from .chain import build_rag_chain

        models = models or build_models(self.settings.model)
        return build_rag_chain(self.retriever(k, mmr), models.llm)

    def agent_graph(self, k: int | None = None, mmr: bool = False, models=None, checkpointer=None):
        from .agent import build_models
        from .graph import build_graph

        models = models or build_models(self.settings.model)
        return build_graph(
            models,
            self.retriever(k, mmr),
            topics=self.topics(),
            max_retries=self.settings.max_retries,
            checkpointer=checkpointer,
        )
