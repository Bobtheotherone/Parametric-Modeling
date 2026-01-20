from __future__ import annotations

from pathlib import Path

from formula_foundry.substrate import ArtifactManifest, ArtifactStore, canonical_json_dumps, sha256_bytes


def test_store_add_bytes_content_addressed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    entry = store.add_bytes("payload.bin", b"hello")

    expected_digest = sha256_bytes(b"hello")

    assert entry.digest == expected_digest
    assert entry.size_bytes == 5
    assert store.object_path(expected_digest).exists()
    assert store.verify_digest(expected_digest)


def test_transaction_writes_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    src_path = tmp_path / "input.txt"
    src_path.write_text("hi", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    tx = store.start_transaction(manifest_path)
    entry_b = tx.add_bytes("b.bin", b"payload")
    entry_a = tx.add_file("a.txt", src_path)
    manifest = tx.commit()

    assert manifest.artifacts[0].path == "a.txt"
    assert manifest.artifacts[1].path == "b.bin"

    expected_payload = ArtifactManifest.from_entries([entry_b, entry_a]).to_dict()
    assert manifest_path.read_text(encoding="utf-8") == f"{canonical_json_dumps(expected_payload)}\n"


def test_verify_entry_detects_mismatch(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    entry = store.add_bytes("payload.bin", b"hello")
    store.object_path(entry.digest).write_bytes(b"corrupt")

    assert not store.verify_entry(entry)
