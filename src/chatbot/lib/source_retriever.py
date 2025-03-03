import collections
from typing import List
from langchain.schema import Document
from lib.errors_handler import traceback_no_record_found_in_sql


def list_top_k_sources(source_documents: List[Document], k=3) -> str:

    if not source_documents:
        traceback_no_record_found_in_sql()
        return ""

    sources = [
        f'[{source_document.metadata.get("title", source_document.metadata.get("focus_area", "Untitled"))}]({source_document.metadata.get("source", "#")})'
        for source_document in source_documents
    ]

    if sources:
        k = min(k, len(sources))
        distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
        distinct_sources_str = "  \n- ".join(distinct_sources)
        return f"Source(s):  \n- {distinct_sources_str}"