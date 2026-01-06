import os
os.environ.setdefault("RANX_DISABLE_NUMBA", "1")  # Disable ranx's numba backend early
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")   # Hard-disable numba JIT globally
# Limit BLAS/OMP threads to avoid native segfaults / oversubscription at scale
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import sys
from typing import Any, Dict, List

import yaml
import numpy as np
from tqdm.auto import tqdm

from .utils import (
    set_seed,
    get_device,
    setup_logging,
    set_warnings_policy,
    ensure_dirs,
    save_config_snapshot,
    run_id,
    system_info,
    iso_pair,
    write_json,
    result_filename,
    write_visualization_json,
)
from .data_loader import DataLoader
from .pair_constructor import Pair, build_retrieval_corpus, build_positive_pairs, mine_hard_negatives
from .embeddings import EmbeddingBackbone
from .retrieval import run_dense_retrieval
from .metrics import classification_metrics
from .classification import build_features, train_classifier, evaluate_classifier
from .splits import random_split, leave_one_language_out, filter_by_shot_types
from .bm25 import BM25Retriever


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)

    # Strict warnings policy
    set_warnings_policy(cfg.get("logging", {}).get("warnings_as_errors", True))

    out_cfg = cfg.get("output", {})
    results_dir = out_cfg.get("results_dir", "results")
    vis_dir = out_cfg.get("visualizations_dir", "visualizations")
    logs_dir = out_cfg.get("logs_dir", "results/logs")
    ensure_dirs(results_dir, vis_dir, logs_dir)

    logger = setup_logging(logs_dir, cfg.get("logging", {}).get("level", "INFO"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("device", "auto"))
    logger.info(f"Device: {device}")
    logger.info(f"Config path: {config_path}")

    # Load data
    data_dir = os.path.join(os.getcwd(), "data")
    dl = DataLoader(data_dir=data_dir, logs_dir=logs_dir)
    dl.load()

    languages = cfg.get("languages") or dl.languages()
    language_pairs = cfg.get("language_pairs") or []
    shot_types = (
        (cfg.get("splits", {}) or {}).get("filters", {}) or {}
    ).get("shot_types", [])

    logger.info(f"Languages detected: {languages}")

    tasks: List[str] = cfg.get("tasks", [])
    backbones: List[str] = cfg.get("backbones", [])
    batch_size = int(cfg.get("batch_size", 64))

    prompts = cfg.get("embedding_prompts", {})

    # Metadata snapshot
    meta = {"config_path": config_path, "seed": seed, "system": system_info()}

    # T1: Monolingual retrieval
    if "t1_mono_retrieval" in tasks:
        for bb_name in backbones:
            bb = EmbeddingBackbone(bb_name, device=device, prompts=prompts)
            # collect per-script recall@1 for aggregation
            script_recalls: Dict[str, List[float]] = {}
            for lang in tqdm(languages, desc="T1 languages"):
                recs = dl.by_lang.get(lang, [])
                if not recs:
                    continue
                split_tag = "dev"
                recs_eval = filter_by_shot_types(recs, shot_types) if shot_types else recs
                if not recs_eval:
                    continue
                qids, queries, dids, docs = build_retrieval_corpus(recs_eval, mode="persona->instruction")
                rid = run_id("t1_mono_retrieval")
                run_dir = os.path.join(results_dir, "runs", "t1_mono_retrieval", rid)
                os.makedirs(run_dir, exist_ok=True)
                save_config_snapshot(run_dir, cfg)
                metrics = run_dense_retrieval(
                    bb,
                    qids,
                    queries,
                    dids,
                    docs,
                    batch_size=batch_size,
                    top_k_values=cfg.get("top_k", [1, 5, 10]),
                    run_dir=run_dir,
                )
                out_name = result_filename("t1_mono_retrieval", bb_name, lang, split_tag, meta['system']['time_utc'])
                write_json(os.path.join(results_dir, out_name), {
                    "task": "t1_mono_retrieval",
                    "backbone": bb_name,
                    "lang_or_pair": lang,
                    "split_tag": split_tag,
                    "timestamp": meta['system']['time_utc'],
                    "metrics": metrics,
                    "meta": meta,
                })
                # Visualization: recall@k lineplot per language
                viz_path = os.path.join(
                    vis_dir,
                    f"recall_at_k-lineplot-{bb_name}-t1_mono_retrieval-{lang}.json",
                )
                write_visualization_json(
                    viz_path,
                    metric="recall",
                    axes={"x": "k", "y": "recall"},
                    data={"k": cfg.get("top_k", [1, 5, 10]), "recall": [metrics["recall"].get(k) for k in cfg.get("top_k", [1, 5, 10])]},
                    notes="Monolingual persona->instruction",
                )
                # Collect per-script recall@1
                script = recs[0].script
                r1 = float(metrics["recall"].get(1, 0.0))
                script_recalls.setdefault(script, []).append(r1)

            # BM25 baseline (optional, same corpus) for persona->instruction
            if cfg.get("bm25"):
                for lang in tqdm(languages, desc="T1 BM25 languages"):
                    recs = dl.by_lang.get(lang, [])
                    if not recs:
                        continue
                    recs_eval = filter_by_shot_types(recs, shot_types) if shot_types else recs
                    if not recs_eval:
                        continue
                    qids, queries, dids, docs = build_retrieval_corpus(recs_eval, mode="persona->instruction")
                    bm25 = BM25Retriever(docs)
                    top_k = max(cfg.get("top_k", [1, 5, 10]))
                    ranked = bm25.search(queries, top_k=top_k)
                    # Convert to ranx run
                    run = {qid: {dids[idx]: float(score) for idx, score in r} for qid, r in zip(qids, ranked)}
                    from .retrieval import build_qrels
                    qrels = build_qrels(qids, dids)
                    from .metrics import retrieval_metrics
                    metrics = retrieval_metrics(qrels, run, ks=cfg.get("top_k", [1, 5, 10]))
                    out_name = result_filename("t1_mono_retrieval", "bm25", lang, "dev", meta['system']['time_utc'])
                    write_json(os.path.join(results_dir, out_name), {
                        "task": "t1_mono_retrieval",
                        "backbone": "bm25",
                        "lang_or_pair": lang,
                        "split_tag": "dev",
                        "timestamp": meta['system']['time_utc'],
                        "metrics": metrics,
                        "meta": meta,
                    })

            # Script-group aggregation visualization (recall@1)
            if script_recalls:
                script_means = {s: float(np.mean(v)) for s, v in script_recalls.items()}
                viz_path = os.path.join(
                    vis_dir,
                    f"script_gap-bar-{bb_name}-dev.json",
                )
                write_visualization_json(
                    viz_path,
                    metric="recall@1",
                    axes={"x": "script", "y": "recall@1"},
                    data=script_means,
                    notes="Average recall@1 by script (monolingual persona->instruction)",
                )

    # T2: Cross-lingual retrieval (persona in LA, instruction in LB)
    if "t2_xling_retrieval" in tasks:
        pairs = language_pairs or [iso_pair(a, b) for a in languages for b in languages if a != b]
        for bb_name in backbones:
            bb = EmbeddingBackbone(bb_name, device=device, prompts=prompts)
            heatmap: Dict[str, Dict[str, float]] = {}
            from glob import glob
            for pr in tqdm(pairs, desc="T2 lang pairs"):
                la, lb = pr.split("->")
                recs_a = dl.by_lang.get(la, [])
                recs_b = dl.by_lang.get(lb, [])
                if not recs_a or not recs_b:
                    continue
                # Resume support: skip pair if results already exist
                pattern = os.path.join(results_dir, result_filename("t2_xling_retrieval", bb_name, f"{la}->{lb}", "dev", "*"))
                try:
                    existing = glob(pattern)
                except Exception:
                    existing = []
                if existing:
                    continue
                # Align by id: queries from A, docs from B (matching ids are positives)
                id_to_b = {r.id: r for r in recs_b}
                qids: List[str] = []
                queries: List[str] = []
                dids: List[str] = []
                docs: List[str] = []
                # Apply shot-type filters if provided
                recs_a_eval = filter_by_shot_types(recs_a, shot_types) if shot_types else recs_a
                for ra in recs_a_eval:
                    if ra.id in id_to_b:
                        qids.append(ra.id)
                        queries.append(ra.persona)
                        dids.append(id_to_b[ra.id].id)
                        docs.append(id_to_b[ra.id].instruction)
                if not qids:
                    continue
                rid = run_id("t2_xling_retrieval")
                run_dir = os.path.join(results_dir, "runs", "t2_xling_retrieval", rid)
                os.makedirs(run_dir, exist_ok=True)
                save_config_snapshot(run_dir, cfg)
                metrics = run_dense_retrieval(
                    bb,
                    qids,
                    queries,
                    dids,
                    docs,
                    batch_size=batch_size,
                    top_k_values=cfg.get("top_k", [1, 5, 10]),
                    run_dir=run_dir,
                )
                out_name = result_filename("t2_xling_retrieval", bb_name, f"{la}->{lb}", "dev", meta['system']['time_utc'])
                write_json(os.path.join(results_dir, out_name), {
                    "task": "t2_xling_retrieval",
                    "backbone": bb_name,
                    "lang_or_pair": f"{la}->{lb}",
                    "split_tag": "dev",
                    "timestamp": meta['system']['time_utc'],
                    "metrics": metrics,
                    "meta": meta,
                })
                heatmap.setdefault(la, {})[lb] = float(metrics["recall"].get(1, 0.0))

            # Save heatmap visualization JSON (recall@1)
            viz_path = os.path.join(
                vis_dir,
                f"lang_pair_matrix-heatmap-{bb_name}-dev.json",
            )
            write_visualization_json(
                viz_path,
                metric="recall@1",
                axes={"x": "source_lang", "y": "target_lang"},
                data=heatmap,
                notes="Persona->Instruction cross-lingual recall@1",
            )

    # T3: Reverse retrieval (instruction -> persona) mono and xling with same langs/pairs
    if "t3_reverse_retrieval" in tasks:
        for bb_name in backbones:
            bb = EmbeddingBackbone(bb_name, device=device, prompts=prompts)
            # Monolingual
            for lang in tqdm(languages, desc="T3 languages"):
                recs = dl.by_lang.get(lang, [])
                if not recs:
                    continue
                # Resume support: skip if monolingual T3 result exists
                from glob import glob as _glob_t3mono
                pattern = os.path.join(results_dir, result_filename("t3_reverse_retrieval", bb_name, lang, "dev", "*"))
                try:
                    existing = _glob_t3mono(pattern)
                except Exception:
                    existing = []
                if existing:
                    continue
                recs_eval = filter_by_shot_types(recs, shot_types) if shot_types else recs
                if not recs_eval:
                    continue
                qids, queries, dids, docs = build_retrieval_corpus(recs_eval, mode="instruction->persona")
                rid = run_id("t3_reverse_retrieval")
                run_dir = os.path.join(results_dir, "runs", "t3_reverse_retrieval", rid)
                os.makedirs(run_dir, exist_ok=True)
                save_config_snapshot(run_dir, cfg)
                metrics = run_dense_retrieval(bb, qids, queries, dids, docs, batch_size, cfg.get("top_k", [1, 5, 10]), run_dir)
                out_name = result_filename("t3_reverse_retrieval", bb_name, lang, "dev", meta['system']['time_utc'])
                write_json(os.path.join(results_dir, out_name), {
                    "task": "t3_reverse_retrieval",
                    "backbone": bb_name,
                    "lang_or_pair": lang,
                    "split_tag": "dev",
                    "timestamp": meta['system']['time_utc'],
                    "metrics": metrics,
                    "meta": meta,
                })
                viz_path = os.path.join(
                    vis_dir,
                    f"recall_at_k-lineplot-{bb_name}-t3_reverse_retrieval-{lang}.json",
                )
                write_visualization_json(
                    viz_path,
                    metric="recall",
                    axes={"x": "k", "y": "recall"},
                    data={"k": cfg.get("top_k", [1, 5, 10]), "recall": [metrics["recall"].get(k) for k in cfg.get("top_k", [1, 5, 10])]},
                    notes="Monolingual instruction->persona",
                )

            # Cross-lingual
            pairs = language_pairs or [iso_pair(a, b) for a in languages for b in languages if a != b]
            heatmap: Dict[str, Dict[str, float]] = {}
            for pr in pairs:
                la, lb = pr.split("->")
                recs_a = dl.by_lang.get(la, [])
                recs_b = dl.by_lang.get(lb, [])
                if not recs_a or not recs_b:
                    continue
                # Resume support: skip if cross-lingual T3 result exists
                from glob import glob as _glob_t3x
                pattern = os.path.join(results_dir, result_filename("t3_reverse_retrieval", bb_name, f"{la}->{lb}", "dev", "*"))
                try:
                    existing = _glob_t3x(pattern)
                except Exception:
                    existing = []
                if existing:
                    continue
                id_to_b = {r.id: r for r in recs_b}
                qids: List[str] = []
                queries: List[str] = []
                dids: List[str] = []
                docs: List[str] = []
                recs_a_eval = filter_by_shot_types(recs_a, shot_types) if shot_types else recs_a
                for ra in recs_a_eval:
                    if ra.id in id_to_b:
                        qids.append(ra.id)
                        queries.append(ra.instruction)
                        dids.append(id_to_b[ra.id].id)
                        docs.append(id_to_b[ra.id].persona)
                if not qids:
                    continue
                rid = run_id("t3_reverse_retrieval")
                run_dir = os.path.join(results_dir, "runs", "t3_reverse_retrieval", rid)
                os.makedirs(run_dir, exist_ok=True)
                metrics = run_dense_retrieval(bb, qids, queries, dids, docs, batch_size, cfg.get("top_k", [1, 5, 10]), run_dir)
                out_name = result_filename("t3_reverse_retrieval", bb_name, f"{la}->{lb}", "dev", meta['system']['time_utc'])
                write_json(os.path.join(results_dir, out_name), {
                    "task": "t3_reverse_retrieval",
                    "backbone": bb_name,
                    "lang_or_pair": f"{la}->{lb}",
                    "split_tag": "dev",
                    "timestamp": meta['system']['time_utc'],
                    "metrics": metrics,
                    "meta": meta,
                })
                heatmap.setdefault(la, {})[lb] = float(metrics["recall"].get(1, 0.0))

            viz_path = os.path.join(
                vis_dir,
                f"lang_pair_matrix-heatmap-{bb_name}-dev-reverse.json",
            )
            write_visualization_json(
                viz_path,
                metric="recall@1",
                axes={"x": "source_lang", "y": "target_lang"},
                data=heatmap,
                notes="Instruction->Persona cross-lingual recall@1",
            )

    # T4: Compatibility Classification with calibration and ECE
    if "t4_classification" in tasks:
        clf_cfg = cfg.get("classification", {})
        estimator = clf_cfg.get("estimator", "logistic_regression")
        params = clf_cfg.get("params", {})
        calibration = clf_cfg.get("calibration", {"method": "temperature", "bins": 15})

        for bb_name in backbones:
            bb = EmbeddingBackbone(bb_name, device=device, prompts=prompts)

            # Prepare data (all selected languages)
            recs = []
            for lang in languages:
                recs.extend(dl.by_lang.get(lang, []))
            if not recs:
                continue

            # Pairs: positives + hard negatives
            logger.info("Building positive pairs and mining hard negatives ...")
            pos_pairs = build_positive_pairs(recs)
            hn_cfg = cfg.get("hard_negatives", {})
            method = str(hn_cfg.get("method", "string")).lower()
            max_negs = int(hn_cfg.get("max_negatives_per_query", 10))
            if method == "embedding":
                # Fast mining via instruction embeddings per language
                sim_threshold = float(hn_cfg.get("sim_threshold", 0.5))
                neg_pairs: List[Pair] = []  # type: ignore[name-defined]
                # Group by language
                by_lang: Dict[str, List[Any]] = {}
                for r in recs:
                    by_lang.setdefault(r.lang, []).append(r)
                from math import inf
                for lang, recs_lang in tqdm(by_lang.items(), desc="Mining HN (emb, langs)"):
                    if not recs_lang:
                        continue
                    # Encode all instructions for this language once
                    instr_texts = [r.instruction for r in recs_lang]
                    d_emb = bb.encode(instr_texts, batch_size=batch_size, is_query=False)
                    n = len(recs_lang)
                    # Batched self-similarities; reuse doc embs for queries
                    for start in tqdm(range(0, n, batch_size), desc=f"HN-emb {lang} queries"):
                        end = min(start + batch_size, n)
                        q_emb = d_emb[start:end]
                        scores = np.matmul(q_emb, d_emb.T)
                        # Exclude self-matches by setting diagonal segment to -inf
                        for i in range(end - start):
                            scores[i, start + i] = -inf
                        # For each query row select top candidates
                        for i_row, r in enumerate(recs_lang[start:end]):
                            row = scores[i_row]
                            # Filter by threshold
                            if max_negs < n - 1:
                                idx = np.argpartition(-row, max_negs)[:max_negs]
                            else:
                                idx = np.arange(n)
                            # Remove any below threshold
                            keep = [j for j in idx if row[j] >= sim_threshold]
                            if not keep:
                                continue
                            # Order by score desc
                            order = np.array(keep)[np.argsort(-row[keep])]
                            for j in order[:max_negs]:
                                neg = recs_lang[j]
                                neg_pairs.append(Pair(q_id=r.id, d_id=neg.id, q_text=r.persona, d_text=neg.instruction, label=0, lang=lang))  # type: ignore[name-defined]
            else:
                # Default string-based mining (RapidFuzz)
                neg_pairs = mine_hard_negatives(
                    recs,
                    threshold=int(hn_cfg.get("threshold", 70)),
                    max_negatives_per_query=max_negs,
                )
            # Optional caps/sampling to keep training size manageable
            global_cap = hn_cfg.get("global_max_neg_pairs")
            sample_frac = hn_cfg.get("neg_sample_frac")
            if isinstance(global_cap, int) and global_cap > 0 and len(neg_pairs) > global_cap:
                rng_local = np.random.RandomState(seed)
                idx = rng_local.choice(len(neg_pairs), size=global_cap, replace=False)
                neg_pairs = [neg_pairs[i] for i in idx]
                logger.info(f"Applied global_max_neg_pairs={global_cap}; negatives now {len(neg_pairs)}")
            elif isinstance(sample_frac, (float, int)) and 0.0 < float(sample_frac) < 1.0:
                rng_local = np.random.RandomState(seed)
                k = max(1, int(float(sample_frac) * len(neg_pairs)))
                idx = rng_local.choice(len(neg_pairs), size=k, replace=False)
                neg_pairs = [neg_pairs[i] for i in idx]
                logger.info(f"Applied neg_sample_frac={sample_frac}; negatives now {len(neg_pairs)}")

            pairs = pos_pairs + neg_pairs
            logger.info(f"Total pairs: {len(pairs)} | positives: {len(pos_pairs)} | hard negatives: {len(neg_pairs)}")

            # Deduplicate texts to avoid repeated encoding and reduce memory
            logger.info("Encoding unique texts (queries/docs) ...")
            unique_q = {}
            unique_d = {}
            for p in pairs:
                if p.q_text not in unique_q:
                    unique_q[p.q_text] = None
                if p.d_text not in unique_d:
                    unique_d[p.d_text] = None
            uq_list = list(unique_q.keys())
            ud_list = list(unique_d.keys())

            # Encode with progress (batching handled inside EmbeddingBackbone)
            q_vecs = bb.encode(uq_list, batch_size=batch_size, is_query=True)
            d_vecs = bb.encode(ud_list, batch_size=batch_size, is_query=False)
            for i, t in enumerate(uq_list):
                unique_q[t] = q_vecs[i]
            for i, t in enumerate(ud_list):
                unique_d[t] = d_vecs[i]

            # Materialize pair embeddings (with tqdm)
            pairs_emb: List[tuple] = []
            for p in tqdm(pairs, desc="Building features"):
                pairs_emb.append((unique_q[p.q_text], unique_d[p.d_text], p.label))

            # Split (stratified on label to avoid single-class splits)
            rng = np.random.RandomState(seed)
            y_all = np.array([pairs_emb[i][2] for i in range(len(pairs_emb))], dtype=int)
            pos_idx = np.where(y_all == 1)[0]
            neg_idx = np.where(y_all == 0)[0]
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            def split_indices(arr, fracs):
                n = len(arr)
                n_train = int(fracs[0] * n)
                n_dev = int(fracs[1] * n)
                train = arr[:n_train]
                dev = arr[n_train:n_train + n_dev]
                test = arr[n_train + n_dev:]
                return train, dev, test
            fracs = (0.7, 0.1)  # train, dev (rest is test)
            p_tr, p_dev, p_te = split_indices(pos_idx, fracs)
            n_tr, n_dev, n_te = split_indices(neg_idx, fracs)
            train_idx = np.concatenate([p_tr, n_tr])
            dev_idx = np.concatenate([p_dev, n_dev])
            test_idx = np.concatenate([p_te, n_te])
            rng.shuffle(train_idx); rng.shuffle(dev_idx); rng.shuffle(test_idx)

            X_train, y_train = build_features([pairs_emb[i] for i in train_idx])
            X_dev, y_dev = build_features([pairs_emb[i] for i in dev_idx])
            X_test, y_test = build_features([pairs_emb[i] for i in test_idx])

            model = train_classifier(X_train, y_train, estimator=estimator, params=params)
            # Evaluate on dev and test
            dev_eval = evaluate_classifier(model, X_dev, y_dev, calibration)
            test_eval = evaluate_classifier(model, X_test, y_test, calibration)

            rid = run_id("t4_classification")
            run_dir = os.path.join(results_dir, "runs", "t4_classification", rid)
            os.makedirs(run_dir, exist_ok=True)
            save_config_snapshot(run_dir, cfg)

            # Save predictions on test
            # For simplicity, compute calibrated probs again here
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                dec = model.decision_function(X_test)
                probs = 1.0 / (1.0 + np.exp(-dec))
            from .classification import calibrate_probs
            probs_cal = calibrate_probs(probs.reshape(-1, 1), y_test, method=calibration.get("method", "temperature"), bins=int(calibration.get("bins", 15))).reshape(-1)
            preds_rows = [
                {"id": int(i), "y_true": int(y_test[i]), "prob": float(probs[i]), "calibrated_prob": float(probs_cal[i])}
                for i in range(len(y_test))
            ]
            from .utils import write_jsonl
            write_jsonl(os.path.join(run_dir, "predictions.jsonl"), preds_rows)
            write_json(os.path.join(run_dir, "metrics.json"), {"dev": dev_eval, "test": test_eval})

            out_name = result_filename("t4_classification", bb_name, "all", "dev_test", meta['system']['time_utc'])
            write_json(os.path.join(results_dir, out_name), {
                "task": "t4_classification",
                "backbone": bb_name,
                "lang_or_pair": "all",
                "split_tag": "dev_test",
                "timestamp": meta['system']['time_utc'],
                "metrics": {"dev": dev_eval, "test": test_eval},
                "meta": meta,
            })

            # Visualization: calibration reliability curve (bins)
            bins = int(calibration.get("bins", 15))
            # Build reliability data for calibrated probs
            bin_edges = np.linspace(0.0, 1.0, bins + 1)
            idx = np.digitize(probs_cal, bin_edges) - 1
            rel = []
            for b in range(bins):
                mask = idx == b
                if not np.any(mask):
                    rel.append({"bin": b, "conf": None, "acc": None, "count": 0})
                else:
                    conf = float(probs_cal[mask].mean())
                    acc = float((y_test[mask] == (probs_cal[mask] >= 0.5)).mean())
                    rel.append({"bin": b, "conf": conf, "acc": acc, "count": int(mask.sum())})
            viz_path = os.path.join(
                vis_dir,
                f"calibration-reliability_curve-{bb_name}-dev_test.json",
            )
            write_visualization_json(
                viz_path,
                metric="reliability_curve",
                axes={"x": "confidence", "y": "accuracy"},
                data=rel,
                notes="Temperature-calibrated",
            )

            # Error cohorts for culturally dense entries (indices only)
            culture_ids = [r.id for r in recs if isinstance(r.description, str) and ("culture" in r.description.lower() or "cultural" in r.description.lower())]
            error_cohorts = {"culture_ids": culture_ids[:200]}  # cap for size
            write_visualization_json(
                os.path.join(vis_dir, f"error_cohorts-{bb_name}-dev_test.json"),
                metric="cohorts",
                axes={"id": "record_id"},
                data=error_cohorts,
                notes="Indices/examples for culturally-dense entries (no extra annotations)",
            )

        # LOLO classification path
        lolo_target = ((cfg.get("splits", {}) or {}).get("lolo", {}) or {}).get("target")
        if lolo_target:
            for bb_name in backbones:
                bb = EmbeddingBackbone(bb_name, device=device, prompts=prompts)
                lolo = leave_one_language_out(dl.by_lang, held_out_lang=lolo_target)
                train_recs = lolo.get("train", [])
                test_recs = lolo.get("test", [])
                if not train_recs or not test_recs:
                    continue
                # Build pairs
                hn_cfg = cfg.get("hard_negatives", {})
                train_pairs = build_positive_pairs(train_recs) + mine_hard_negatives(
                    train_recs,
                    threshold=int(hn_cfg.get("threshold", 70)),
                    max_negatives_per_query=int(hn_cfg.get("max_negatives_per_query", 10)),
                )
                test_pairs = build_positive_pairs(test_recs) + mine_hard_negatives(
                    test_recs,
                    threshold=int(hn_cfg.get("threshold", 70)),
                    max_negatives_per_query=int(hn_cfg.get("max_negatives_per_query", 10)),
                )
                # Encode
                q_train = bb.encode([p.q_text for p in train_pairs], batch_size=batch_size, is_query=True)
                d_train = bb.encode([p.d_text for p in train_pairs], batch_size=batch_size, is_query=False)
                q_test = bb.encode([p.q_text for p in test_pairs], batch_size=batch_size, is_query=True)
                d_test = bb.encode([p.d_text for p in test_pairs], batch_size=batch_size, is_query=False)
                X_train, y_train = build_features([(q_train[i], d_train[i], train_pairs[i].label) for i in range(len(train_pairs))])
                X_test, y_test = build_features([(q_test[i], d_test[i], test_pairs[i].label) for i in range(len(test_pairs))])
                model = train_classifier(X_train, y_train, estimator=estimator, params=params)
                test_eval = evaluate_classifier(model, X_test, y_test, calibration)

                rid = run_id("t4_classification_lolo")
                run_dir = os.path.join(results_dir, "runs", "t4_classification", rid)
                os.makedirs(run_dir, exist_ok=True)
                save_config_snapshot(run_dir, cfg)
                # Save test predictions for LOLO
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)[:, 1]
                else:
                    dec = model.decision_function(X_test)
                    probs = 1.0 / (1.0 + np.exp(-dec))
                from .classification import calibrate_probs
                probs_cal = calibrate_probs(probs.reshape(-1, 1), y_test, method=calibration.get("method", "temperature"), bins=int(calibration.get("bins", 15))).reshape(-1)
                from .utils import write_jsonl
                preds_rows = [
                    {"id": int(i), "y_true": int(y_test[i]), "prob": float(probs[i]), "calibrated_prob": float(probs_cal[i])}
                    for i in range(len(y_test))
                ]
                write_jsonl(os.path.join(run_dir, "predictions.jsonl"), preds_rows)
                write_json(os.path.join(run_dir, "metrics.json"), {"lolo_target": lolo_target, "test": test_eval})
                out_name = result_filename("t4_classification", bb_name, lolo_target, f"lolo_{lolo_target}", meta['system']['time_utc'])
                write_json(os.path.join(results_dir, out_name), {
                    "task": "t4_classification",
                    "backbone": bb_name,
                    "lang_or_pair": lolo_target,
                    "split_tag": f"lolo_{lolo_target}",
                    "timestamp": meta['system']['time_utc'],
                    "metrics": {"test": test_eval},
                    "meta": meta,
                })

    # T5: Hard negatives mining report
    if "t5_hardneg_eval" in tasks:
        hn_cfg = cfg.get("hard_negatives", {})
        threshold=int(hn_cfg.get("threshold", 70))
        max_negs=int(hn_cfg.get("max_negatives_per_query", 10))
        report: Dict[str, Any] = {"threshold": threshold, "max_negs": max_negs, "per_lang": {}}
        for lang in tqdm(languages, desc="T5 languages"):
            recs = dl.by_lang.get(lang, [])
            negs = mine_hard_negatives(recs, threshold=threshold, max_negatives_per_query=max_negs)
            report["per_lang"][lang] = {"num_records": len(recs), "num_neg_pairs": len(negs)}
        out_name = result_filename("t5_hardneg_eval", "n/a", "all", "dev", meta['system']['time_utc'])
        write_json(os.path.join(results_dir, out_name), {
            "task": "t5_hardneg_eval",
            "backbone": None,
            "lang_or_pair": "all",
            "split_tag": "dev",
            "timestamp": meta['system']['time_utc'],
            "metrics": report,
            "meta": meta,
        })

    # Note: Other tasks (T2, T3, T4, T5, LOLO) to be implemented in subsequent steps.


if __name__ == "__main__":
    # Allow running via: python -m src.runner [config.yaml]
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg_path)
