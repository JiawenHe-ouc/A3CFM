#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict
import torch
import gradio as gr


def patch_gradio_json_schema_bool_bug() -> None:
    try:
        import gradio_client.utils as client_utils
    except Exception:
        return

    if getattr(client_utils, "_MAFUSION_BOOL_SCHEMA_PATCHED", False):
        return

    old_private = getattr(client_utils, "_json_schema_to_python_type", None)
    old_public = getattr(client_utils, "json_schema_to_python_type", None)

    if old_private is not None:
        def safe_private(schema, defs=None):  
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            if schema is None:
                return "Any"
            return old_private(schema, defs)
        client_utils._json_schema_to_python_type = safe_private

    if old_public is not None:
        def safe_public(schema):  
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            if schema is None:
                return "Any"
            return old_public(schema)
        client_utils.json_schema_to_python_type = safe_public

    client_utils._MAFUSION_BOOL_SCHEMA_PATCHED = True


patch_gradio_json_schema_bool_bug()

from mme_rag.config import load_yaml
from mme_rag.inference import (
    load_face_file,
    load_multimodal_model,
    load_physio_file,
    preprocess_gradio_audio,
    run_emotion_inference,
)
from mme_rag.rag_engine import MaritimeRAGEngine
from mme_rag.reporting import LlamaReporter
from mme_rag.sensevoice import SenseVoiceASR


def str2bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def patch_local_proxy_env() -> None:
    local_hosts = "localhost,127.0.0.1,0.0.0.0,::1"
    for key in ("NO_PROXY", "no_proxy"):
        old = os.environ.get(key, "")
        if old:
            parts = [p.strip() for p in old.split(",") if p.strip()]
            for item in local_hosts.split(","):
                if item not in parts:
                    parts.append(item)
            os.environ[key] = ",".join(parts)
        else:
            os.environ[key] = local_hosts


def choose_checkpoint(cfg: Dict[str, Any], cli_checkpoint: str | None) -> str:
    candidates = []
    if cli_checkpoint:
        candidates.append(cli_checkpoint)
    inf = cfg.get("inference", {})
    if inf.get("checkpoint"):
        candidates.append(inf["checkpoint"])
    candidates.extend(inf.get("fallback_checkpoints", []))
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("No checkpoint found. Tried:\n" + "\n".join(map(str, candidates)))


class AppState:
    def __init__(self, cfg: Dict[str, Any], checkpoint: str):
        self.cfg = cfg
        paths = cfg.get("paths", {})
        app_cfg = cfg.get("app", {})
        inf_cfg = cfg.get("inference", {})

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index if self.device.index is not None else 0)
        self.llm_device = torch.device(
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else self.device
        )
        if self.llm_device.type == "cuda":
            torch.cuda.set_device(self.llm_device.index if self.llm_device.index is not None else 0)

        # self.model, self.ckpt_cfg = load_multimodal_model(checkpoint)
        # self.device = next(self.model.parameters()).device

        self.model, self.ckpt_cfg = load_multimodal_model(
            checkpoint,
            device=self.device,
        )

        self.asr = SenseVoiceASR(
            paths.get("sense_voice_path", "./iic/SenseVoiceSmall"),
            offline=str2bool(app_cfg.get("offline", True), default=True),
        )
        self.rag = MaritimeRAGEngine(
            embed_model_path=paths.get("embed_model_path", "./iic/paraphrase-multilingual-MiniLM-L12-v2"),
            db_path=paths.get("chroma_db_path", "./maritime_chroma_db"),
            base_knowledge_json=paths.get("base_knowledge_json", "./baseKnowledge.json"),
            custom_docs_dir=paths.get("custom_docs_dir", "./maritime_docs"),
        )
        self.reporter = LlamaReporter(
            model_path=paths.get("llama_model_path", "../meta-llama/llava-llama-3-8b"),
            load_4bit=str2bool(app_cfg.get("load_llama_4bit", True), default=True),
            allow_template_fallback=str2bool(app_cfg.get("allow_template_report_without_llama", False), default=False),
            offline=str2bool(app_cfg.get("offline", True), default=True),
            llm_device=self.llm_device,
        )

        self.sample_rate = int(inf_cfg.get("sample_rate", 16000))
        self.clip_seconds = float(inf_cfg.get("clip_seconds", 6.0))
        self.rag_top_k = int(inf_cfg.get("rag_top_k", 4))
        self.image_size = int(inf_cfg.get("image_size", 112))


def build_query(asr_text: str, pred: Dict[str, Any], role: str, task: str, background: str) -> str:
    rel = (
        f"audio_rel:{pred.get('audio_reliability')} "
        f"physio_rel:{pred.get('physio_reliability')} "
        f"face_rel:{pred.get('face_reliability')}"
    )
    return (
        f"{role} {task} {background} {asr_text} "
        f"emotion:{pred.get('emotion')} stress:{pred.get('stress_score')} "
        f"risk:{pred.get('risk_level')} {rel} affect:{pred.get('affect_latent_summary')}"
    )


import time
import traceback

def make_infer_fn(state: AppState):
    def infer(audio_input, physio_file, face_file, language, background, role, task, rag_top_k):
        t0 = time.time()

        def log(msg):
            print(f"[infer] {msg} | elapsed={time.time() - t0:.2f}s", flush=True)

        try:
            log("start")

            log("preprocess audio")
            audio_np, audio_tensor = preprocess_gradio_audio(
                audio_input,
                state.sample_rate,
                state.clip_seconds
            )
            log(f"audio preprocessed: audio_np={getattr(audio_np, 'shape', None)}, audio_tensor={getattr(audio_tensor, 'shape', None)}")

            log("ASR start")
            asr_result = (
                state.asr.transcribe(audio_np, language=language or "auto")
                if audio_np.size else {"raw": "", "text": ""}
            )
            log(f"ASR done: text={asr_result.get('text', '')[:80]}")

            log("load physio start")
            first = state.model.physio_encoder.features[0].net[0]
            expected_channels = int(first.weight.shape[1])
            expected_steps = None
            data_cfg = state.ckpt_cfg.get("data", {})
            if "target_hz" in data_cfg and "window_seconds" in data_cfg:
                expected_steps = int(round(float(data_cfg["target_hz"]) * float(data_cfg["window_seconds"])))

            physio_tensor = load_physio_file(
                physio_file.name if physio_file else None,
                expected_channels=expected_channels,
                expected_steps=expected_steps,
            )
            log(f"load physio done: {None if physio_tensor is None else tuple(physio_tensor.shape)}")

            log("load face start")
            face_tensor = load_face_file(
                face_file.name if face_file else None,
                image_size=state.image_size
            )
            log(f"load face done: {None if face_tensor is None else tuple(face_tensor.shape)}")

            prompt_text = (
                f"Role: {role}. Task: {task}. Background: {background}. "
                f"Instruction: assess seafarer affect, stress, fatigue, and maritime safety risk."
            )

            log("emotion inference start")
            pred = run_emotion_inference(
                state.model,
                audio_tensor,
                physio_tensor,
                face_tensor,
                device=state.device,
                prompt_text=prompt_text
            )
            log("emotion inference done")

            log("RAG query build")
            query = build_query(asr_result["text"], pred, role, task, background)

            log("RAG retrieve start")
            retrieved = state.rag.retrieve_affect_aware(
                query,
                affect=pred,
                top_k=int(rag_top_k)
            )
            log(f"RAG retrieve done: {len(retrieved)} results")

            log("RAG format start")
            rag_context = state.rag.format_context(retrieved)
            log("RAG format done")

            log("LLM report generation start")
            import torch

            log("LLM diagnostics")

            reporter_model = getattr(state.reporter, "model", None)

            print(
                "[LLM] model type:",
                type(reporter_model),
                flush=True,
            )

            if reporter_model is not None:
                print(
                    "[LLM] hf_device_map:",
                    getattr(reporter_model, "hf_device_map", None),
                    flush=True,
                )

            for i in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                print(
                    f"[GPU {i}] "
                    f"free={free_mem / 1024**3:.2f} GB, "
                    f"total={total_mem / 1024**3:.2f} GB, "
                    f"allocated={torch.cuda.memory_allocated(i)/1024**3:.2f} GB, "
                    f"reserved={torch.cuda.memory_reserved(i)/1024**3:.2f} GB",
                    flush=True,
                )
            report = state.reporter.generate(
                asr_result["text"],
                pred,
                rag_context,
                role,
                task,
                background
            )
            log("LLM report generation done")

            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            result = {
                "timestamp": timestamp,
                "asr": asr_result,
                "prediction": pred,
                "role": role,
                "task": task,
                "background": background,
                "rag_results": retrieved,
                "report": report,
            }

            out_path = Path(f"multimodal_emotion_result_{timestamp}.json")
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            retrieval_text = "\n\n".join(
                f"[{r['category']}] score={r['score']:.2f}\n{r['text'][:220]}"
                for r in retrieved
            )

            log("finished")
            return (
                asr_result["text"],
                json.dumps(pred, ensure_ascii=False, indent=2),
                retrieval_text,
                report,
                str(out_path),
            )

        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            return (
                "[ERROR]",
                "{}",
                "",
                err,
                "",
            )

    return infer

def launch(cfg: Dict[str, Any], checkpoint: str):
    patch_local_proxy_env()
    patch_gradio_json_schema_bool_bug()

    state = AppState(cfg, checkpoint)
    app_cfg = cfg.get("app", {})
    inf_cfg = cfg.get("inference", {})

    server_name = app_cfg.get("server_name", "0.0.0.0")
    server_port = int(app_cfg.get("server_port", 7860))
    share = str2bool(app_cfg.get("share", False), default=False)

    with gr.Blocks(theme=gr.themes.Soft(), title="Reliability-aware Affective Fusion RAG") as demo:
        gr.Markdown("# Reliability-aware Unpaired Multi-view Affective Fusion + RAG Report")
        gr.Markdown(
            "Audio + physiology + face → shared affect latent → reliability-aware fusion "
            "→ affect-aware retrieval → affect-conditioned LLM report"
        )
        with gr.Row():
            with gr.Column(scale=3):
                audio = gr.Audio(label="Speech audio input", type="numpy")
                physio = gr.File(
                    label="Physiological window (.npy/.csv/.txt, [C,T] or [T,C])",
                    file_types=[".npy", ".csv", ".txt"],
                )
                face = gr.File(
                    label="Facial expression image (.jpg/.png)",
                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                )
                language = gr.Dropdown(
                    ["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                    value="auto",
                    label="SenseVoice language",
                )
                background = gr.Textbox(
                    label="Background / culture",
                    value="Chinese seafarer, international fleet, standardized communication, moderate stress tolerance",
                )
                role = gr.Textbox(label="Role", value="Second officer on watch")
                task = gr.Textbox(label="Task", value="Night navigation in dense traffic and complex sea state")
                rag_top_k = gr.Slider(
                    1,
                    10,
                    value=int(inf_cfg.get("rag_top_k", 4)),
                    step=1,
                    label="RAG top-k",
                )
                btn = gr.Button("Run analysis", variant="primary")
            with gr.Column(scale=4):
                asr_out = gr.Textbox(label="ASR transcript", lines=3)
                pred_out = gr.Textbox(label="Affective state inference", lines=12)
                rag_out = gr.Textbox(label="Affect-aware RAG references", lines=10)
                report_out = gr.Textbox(label="Interpretable report", lines=20)
                file_out = gr.Textbox(label="Result JSON path", lines=1)

        btn.click(
            make_infer_fn(state),
            inputs=[audio, physio, face, language, background, role, task, rag_top_k],
            outputs=[asr_out, pred_out, rag_out, report_out, file_out],
            api_name=False,
        )

    print(f"[INFO] Launching Gradio: server_name={server_name}, server_port={server_port}, share={share}")
    print(f"[INFO] Launching Gradio: server_name={server_name}, server_port={server_port}, share={share}")

    demo.queue(max_size=8, default_concurrency_limit=1)
    try:
        demo.launch(server_name=server_name, server_port=server_port, share=share, show_api=False)
    except TypeError:
        demo.launch(server_name=server_name, server_port=server_port, share=share)
    except ValueError as e:
        msg = str(e)
        if (not share) and ("localhost is not accessible" in msg or "shareable link must be created" in msg):
            print("[WARN] Gradio cannot verify localhost access on this server.")
            print("[WARN] Retrying with share=True. If the machine is offline, use SSH port forwarding instead.")
            try:
                demo.launch(server_name=server_name, server_port=server_port, share=True, show_api=False)
            except TypeError:
                demo.launch(server_name=server_name, server_port=server_port, share=True)
        else:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--no-share", action="store_true", help="Disable public Gradio share link.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    app_cfg = cfg.setdefault("app", {})

    if args.server_name:
        app_cfg["server_name"] = args.server_name
    if args.server_port:
        app_cfg["server_port"] = args.server_port
    if args.share:
        app_cfg["share"] = True
    if args.no_share:
        app_cfg["share"] = False

    ckpt = choose_checkpoint(cfg, args.checkpoint)
    launch(cfg, ckpt)


if __name__ == "__main__":
    main()
