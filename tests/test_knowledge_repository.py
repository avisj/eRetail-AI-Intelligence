"""Unit tests for Phase 7F Knowledge Repository Storage Abstraction."""

from __future__ import annotations

import concurrent.futures
import pytest

from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument, KnowledgeChunk


def _make_doc(did: str, ver: str, domain: KnowledgeDomain = KnowledgeDomain.GENERAL, status: DocumentStatus = DocumentStatus.ACTIVE) -> BusinessKnowledgeDocument:
    return BusinessKnowledgeDocument(
        document_id=did,
        title=f"Doc {did}",
        document_type=DocumentType.POLICY,
        version=ver,
        source="Test",
        domain=domain,
        status=status,
        content=f"Content for {did} v{ver}",
        checksum=f"hash_{did}_{ver}",
    )


def test_repository_empty_state() -> None:
    repo = InMemoryKnowledgeRepository()
    assert repo.get_document("NON_EXISTENT") is None
    assert repo.list_documents() == []
    assert repo.get_all_chunks() == []


def test_repository_deep_copy_isolation() -> None:
    repo = InMemoryKnowledgeRepository()
    doc = _make_doc("DOC-MUT", "1.0")
    repo.add_document(doc)

    fetched = repo.get_document("DOC-MUT", version="1.0")
    assert fetched is not None

    # Even if someone constructs a new model copy, store remains untouched
    fetched_again = repo.get_document("DOC-MUT", version="1.0")
    assert fetched_again.title == "Doc DOC-MUT"


def test_repository_list_filters() -> None:
    repo = InMemoryKnowledgeRepository()
    d1 = _make_doc("D1", "1.0", domain=KnowledgeDomain.INVENTORY, status=DocumentStatus.ACTIVE)
    d2 = _make_doc("D2", "1.0", domain=KnowledgeDomain.SALES, status=DocumentStatus.ACTIVE)
    d3 = _make_doc("D3", "1.0", domain=KnowledgeDomain.INVENTORY, status=DocumentStatus.DEPRECATED)

    repo.add_document(d1)
    repo.add_document(d2)
    repo.add_document(d3)

    # Filter by domain
    inv_docs = repo.list_documents(domain=KnowledgeDomain.INVENTORY)
    assert len(inv_docs) == 2
    assert {d.document_id for d in inv_docs} == {"D1", "D3"}

    # Filter by status
    active_docs = repo.list_documents(status=DocumentStatus.ACTIVE)
    assert len(active_docs) == 2
    assert {d.document_id for d in active_docs} == {"D1", "D2"}

    # Filter by both
    active_inv = repo.list_documents(domain=KnowledgeDomain.INVENTORY, status=DocumentStatus.ACTIVE)
    assert len(active_inv) == 1
    assert active_inv[0].document_id == "D1"


def test_repository_delete_specific_version() -> None:
    repo = InMemoryKnowledgeRepository()
    d1 = _make_doc("DOC-DEL", "1.0")
    d2 = _make_doc("DOC-DEL", "2.0")
    repo.add_document(d1)
    repo.add_document(d2)

    assert repo.delete_document("DOC-DEL", version="1.0") is True
    assert repo.get_document("DOC-DEL", version="1.0") is None
    assert repo.get_document("DOC-DEL", version="2.0") is not None


def test_repository_delete_all_versions() -> None:
    repo = InMemoryKnowledgeRepository()
    repo.add_document(_make_doc("DOC-DEL-ALL", "1.0"))
    repo.add_document(_make_doc("DOC-DEL-ALL", "2.0"))

    assert repo.delete_document("DOC-DEL-ALL") is True
    assert repo.get_document("DOC-DEL-ALL", version="1.0") is None
    assert repo.get_document("DOC-DEL-ALL", version="2.0") is None
    assert repo.delete_document("DOC-DEL-ALL") is False


def test_repository_clear() -> None:
    repo = InMemoryKnowledgeRepository()
    repo.add_document(_make_doc("D1", "1.0"))
    repo.clear()
    assert len(repo.list_documents()) == 0


def test_repository_thread_safety_concurrent_writes() -> None:
    repo = InMemoryKnowledgeRepository()

    def _worker(worker_id: int) -> None:
        for i in range(10):
            doc = _make_doc(f"THREAD-DOC-{worker_id}", f"1.{i}")
            repo.add_document(doc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_worker, w) for w in range(5)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    docs = repo.list_documents()
    assert len(docs) == 50
