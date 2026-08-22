from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


class MaritimeRAGEngine:
    def __init__(self, embed_model_path: str, db_path: str, base_knowledge_json: str, custom_docs_dir: str | None = None, collection_name: str = "maritime_knowledge"):
        if not Path(embed_model_path).exists():
            raise FileNotFoundError(f"Embedding model path not found: {embed_model_path}")
        self.embed_model = SentenceTransformer(embed_model_path)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        if self.collection.count() == 0:
            self._populate_base(base_knowledge_json)
        if custom_docs_dir:
            self.add_documents_from_folder(custom_docs_dir)

    def _load_base(self, path: str) -> List[Dict[str, str]]:
        path = Path(path)
        if not path.exists():
            # Keep app executable for first launch.
            return [
                {"id": "default_fatigue", "category": "疲劳管理", "text": "持续高压力、疲劳或负面情绪会影响注意力、沟通和航行安全，应及时休息、复核任务并上报风险。"},
                {"id": "default_stress", "category": "压力干预", "text": "当船员出现明显压力反应时，应采用短时呼吸调节、任务再分配、同伴支持和必要的医疗/心理支持。"},
            ]
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k in ("knowledge", "data", "items", "maritime_knowledge"):
                if k in data:
                    data = data[k]
                    break
        if not isinstance(data, list):
            raise ValueError("baseKnowledge.json must be list or dict containing knowledge/data/items")
        out = []
        for i, item in enumerate(data):
            out.append({"id": str(item.get("id", f"base_{i}")), "category": str(item.get("category", "未分类")), "text": str(item.get("text", ""))})
        return [x for x in out if len(x["text"]) > 5]

    def _populate_base(self, path: str) -> None:
        items = self._load_base(path)
        texts = [x["text"] for x in items]
        ids = [x["id"] for x in items]
        metas = [{"category": x["category"], "source": Path(path).name} for x in items]
        emb = self.embed_model.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(documents=texts, embeddings=emb, ids=ids, metadatas=metas)

    def add_documents_from_folder(self, folder_path: str) -> int:
        folder = Path(folder_path)
        if not folder.exists():
            return 0
        added = 0
        for p in sorted(folder.glob("*.txt")):
            content = p.read_text(encoding="utf-8")
            paras = [x.strip() for x in content.split("\n\n") if len(x.strip()) > 30]
            if not paras:
                continue
            ids, docs, metas = [], [], []
            for i, para in enumerate(paras):
                doc_id = f"custom__{p.name}__{i}"
                if self.collection.get(ids=[doc_id]).get("ids"):
                    continue
                ids.append(doc_id); docs.append(para); metas.append({"category": "用户自定义", "source": p.name})
            if docs:
                emb = self.embed_model.encode(docs, show_progress_bar=False).tolist()
                self.collection.add(documents=docs, embeddings=emb, ids=ids, metadatas=metas)
                added += len(docs)
        return added

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        n = min(int(top_k), max(1, self.collection.count()))
        q = self.embed_model.encode([query], show_progress_bar=False).tolist()
        res = self.collection.query(query_embeddings=q, n_results=n, include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "category": meta.get("category", "未知"), "source": meta.get("source", "builtin"), "score": round(1.0 - float(dist), 4)})
        return out

    def build_affect_aware_query(self, query: str, affect: Dict[str, Any] | None = None) -> str:
        """Emotion-conditioned retrieval query.

        Roadmap mapping:
            transcription + role/task/background + z_affect(t)
            -> Affect-aware Retriever
        """
        if not affect:
            return query
        emotion = affect.get("emotion", "unknown")
        stress_score = affect.get("stress_score", "unknown")
        risk = affect.get("risk_level", "unknown")
        confidence = affect.get("confidence", "unknown")
        reliability = []
        if affect.get("audio_reliability") is not None:
            reliability.append(f"audio_reliability:{affect.get('audio_reliability')}")
        if affect.get("physio_reliability") is not None:
            reliability.append(f"physio_reliability:{affect.get('physio_reliability')}")
        if affect.get("face_reliability") is not None:
            reliability.append(f"face_reliability:{affect.get('face_reliability')}")
        if affect.get("fusion_modality_weights") is not None:
            reliability.append(f"modality_weights:{affect.get('fusion_modality_weights')}")
        expansion = (
            f" emotion:{emotion} 情绪:{emotion} stress_score:{stress_score} 压力评分:{stress_score} "
            f"risk:{risk} 风险等级:{risk} confidence:{confidence} "
            + " ".join(reliability)
        )
        summary = affect.get("affect_latent_summary") or {}
        if summary:
            expansion += f" affect_latent_norm:{summary.get('l2_norm')} affect_latent_std:{summary.get('std')}"
        return f"{query} {expansion}"

    def retrieve_affect_aware(self, query: str, affect: Dict[str, Any] | None = None, top_k: int = 4) -> List[Dict[str, Any]]:
        return self.retrieve(self.build_affect_aware_query(query, affect), top_k=top_k)

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
        if not results:
            return "（无相关参考知识）"
        return "\n\n".join(f"【参考{i}】[{r['category']}]（相关度:{r['score']:.2f}）\n{r['text']}" for i, r in enumerate(results, 1))
