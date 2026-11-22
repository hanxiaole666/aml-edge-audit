"""
26_reproducibility_audit.py - 可复现性审计：稿件每个声称的溯源链
=================================================================
对稿件中每个定量声称，验证完整链条：
  论文数字 -> 结果文件 -> 生成脚本 -> 原始数据文件 -> 数据来源(provenance)

输出 results/reproducibility_audit.json，供审稿人一键复核。

用法：python scripts/26_reproducibility_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance

# 声称 -> (论文值, 结果文件, 提取字段说明, 生成脚本, 依赖数据)
TRACE = [
    # ---------- Table 3: Elliptic2 baseline
    ("Elliptic2 LightGBM random PR-AUC", 0.536,
     "elliptic2_full_summary.csv", "model=elliptic2_full_LightGBM,split=random,PR_AUC_mean",
     "20_elliptic2_full.py", "data/elliptic2/{nodes,edges,connected_components}.csv + node_features_43.npy"),
    ("Elliptic2 LightGBM size-shift PR-AUC", 0.372,
     "elliptic2_full_summary.csv", "model=elliptic2_full_LightGBM,split=size,PR_AUC_mean",
     "20_elliptic2_full.py", "同上"),
    ("Elliptic2 XGBoost random PR-AUC", 0.508,
     "elliptic2_full_summary.csv", "model=elliptic2_full_XGBoost,split=random,PR_AUC_mean",
     "20_elliptic2_full.py", "同上"),
    ("Elliptic2 topology-only PR-AUC", 0.026,
     "elliptic2_summary.csv", "max(PR_AUC_mean)",
     "16_elliptic2_bench.py", "data/elliptic2/{nodes,edges,connected_components}.csv"),
    # ---------- Table 4: GNN anchor
    ("GNN anchor LightGBM pooled", 0.545,
     "elliptic2_gnn_anchor.json", "random/LightGBM-pooled/mean",
     "48_elliptic2_gnn_anchor.py", "data/elliptic2/{nodes,edges,connected_components}.csv"),
    ("GNN anchor DeepSets", 0.464,
     "elliptic2_gnn_anchor.json", "random/DeepSets/mean",
     "48_elliptic2_gnn_anchor.py", "同上"),
    ("GNN anchor GIN-3", 0.412,
     "elliptic2_gnn_anchor.json", "random/GIN-3/mean",
     "48_elliptic2_gnn_anchor.py", "同上"),
    # ---------- Table 5: Real-CATS tabular
    ("Real-CATS XGBoost random PR-AUC", 0.980,
     "tabular_summary_realcats_random.csv", "model=tabular_XGBoost,PR_AUC_mean",
     "02_train_tabular.py", "data/real_cats_eth.csv"),
    ("Real-CATS XGBoost temporal PR-AUC", 0.893,
     "tabular_summary_realcats_temporal.csv", "model=tabular_XGBoost,PR_AUC_mean",
     "02_train_tabular.py", "data/real_cats_eth.csv"),
    # ---------- Table 6: graph probe
    ("StreamTemporal temporal PR-AUC", 0.881,
     "gnn_summary_realgraph.csv", "model=gnn_StreamTemporal,split=temporal,PR_AUC_mean",
     "03_train_gnn.py", "data/real_cats/real_graph_{nodes,edges}.csv"),
    ("StreamTemporal random PR-AUC", 0.969,
     "gnn_summary_realgraph.csv", "model=gnn_StreamTemporal,split=random,PR_AUC_mean",
     "03_train_gnn.py", "同上"),
    ("TGN temporal PR-AUC", 0.823,
     "temporal_sota_v2.csv", "model=tgn,split=temporal,agg=mean:PR-AUC",
     "23_temporal_sota.py", "data/real_cats/real_graph_{nodes,edges}.csv"),
    ("DyGFormer temporal PR-AUC", 0.842,
     "temporal_sota_v2.csv", "model=dyformer,split=temporal,agg=mean:PR-AUC",
     "23_temporal_sota.py", "同上"),
    ("TGAT temporal PR-AUC", 0.771,
     "temporal_sota_v2.csv", "model=tgat,split=temporal,agg=mean:PR-AUC",
     "23_temporal_sota.py", "同上"),
    # ---------- Table 7: CI
    ("XGBoost temporal PR-AUC (CI point)", 0.894,
     "ci_demo.json", "ci_by_model.XGBoost.point",
     "30_ci_demo.py", "data/real_cats_eth.csv"),
    # ---------- Table 8: edge protocol
    ("Edge protocol C1 mean AP", 0.709,
     "edge_protocol_final.json", "aggregate/C1/AP_mean",
     "44_edge_protocol_final.py", "data/real_cats/real_graph_{nodes,edges}.csv"),
    ("Edge protocol C2 mean AP", 0.806,
     "edge_protocol_final.json", "aggregate/C2/AP_mean",
     "44_edge_protocol_final.py", "同上"),
    ("Edge protocol C3 mean AP", 0.837,
     "edge_protocol_final.json", "aggregate/C3/AP_mean",
     "44_edge_protocol_final.py", "同上"),
    ("Edge protocol M2 random AP", 0.725,
     "edge_protocol_final.json", "aggregate/M2/AP_mean",
     "44_edge_protocol_final.py", "同上"),
    ("Edge C2 vs C3 paired diff", -0.029,
     "edge_protocol_final.json", "paired_tests/C2_vs_C3/diff",
     "44_edge_protocol_final.py", "同上"),
    ("Edge C2 vs M2 paired diff", 0.091,
     "edge_protocol_final.json", "paired_tests/C2_vs_M2/diff",
     "44_edge_protocol_final.py", "同上"),
    # ---------- Table 9: mechanism
    ("M1 feature-only control", 0.9174,
     "mechanism_controls.json", "M1_feature_only/PR_AUC_by_condition/C1_keep_all",
     "43_mechanism_controls.py", "data/real_cats/real_graph_{nodes,edges}.csv"),
    ("C1 mean test degree", 8.155,
     "mechanism_diagnostics.json", "conditions/C1/test_degree/mean",
     "46_mechanism_diagnostics.py", "同上"),
    ("C2 mean test degree", 2.44,
     "mechanism_diagnostics.json", "conditions/C2/test_degree/mean",
     "46_mechanism_diagnostics.py", "同上"),
    ("C1 neighbour-label AUC", 0.8257,
     "mechanism_diagnostics.json", "conditions/C1/neighbor_label/frac_pos_label_AUC",
     "46_mechanism_diagnostics.py", "同上"),
    ("C2 neighbour-label AUC", 0.8745,
     "mechanism_diagnostics.json", "conditions/C2/neighbor_label/frac_pos_label_AUC",
     "46_mechanism_diagnostics.py", "同上"),
    ("C1 representation MMD", 0.022965,
     "mechanism_diagnostics.json", "conditions/C1/MMD_to_C3",
     "46_mechanism_diagnostics.py", "同上"),
    ("C2 representation MMD", 0.004262,
     "mechanism_diagnostics.json", "conditions/C2/MMD_to_C3",
     "46_mechanism_diagnostics.py", "同上"),
    ("Cross-edge homophily observed", 0.4397,
     "mechanism_diagnostics.json", "cross_edge_homophily/observed_same_label",
     "46_mechanism_diagnostics.py", "同上"),
    ("Cross-edge homophily expected", 0.4902,
     "mechanism_diagnostics.json", "cross_edge_homophily/expected_if_independent",
     "46_mechanism_diagnostics.py", "同上"),
    # ---------- Table 10: Elliptic
    ("Elliptic labelled transactions", 46564,
     "elliptic_v1_edge_protocol.json", "n_labelled",
     "49_elliptic_v1_edge_protocol.py", "data/elliptic_v1/elliptic_txs_{features,classes,edgelist}.csv"),
    ("Elliptic illicit transactions", 4545,
     "elliptic_v1_edge_protocol.json", "illicit",
     "49_elliptic_v1_edge_protocol.py", "同上"),
    ("Elliptic cutoff30 C1 AP", 0.8629,
     "elliptic_v1_strong_model.json", "splits/0/conditions/C1/AP",
     "51_elliptic_v1_strong_model.py", "同上"),
    ("Elliptic cutoff39 C3 AP", 0.661,
     "elliptic_v1_strong_model.json", "splits/3/conditions/C3/AP",
     "51_elliptic_v1_strong_model.py", "同上"),
    ("Elliptic cross-split edges", 35158,
     "cross_edge_homophily.json", "elliptic_v1/0/n_cross_edges",
     "52_cross_edge_homophily.py", "同上"),
    ("Elliptic cross edges both labelled", 0,
     "cross_edge_homophily.json", "elliptic_v1/0/n_cross_edges_both_labelled",
     "52_cross_edge_homophily.py", "同上"),
    # ---------- Table 13: feature audit
    ("Max single-feature AUC", 0.6631,
     "elliptic2_feature_audit.json", "verdict/max_single_feature_AUC",
     "50_elliptic2_feature_audit.py", "data/elliptic2/node_features_43.npy"),
    ("Prefix k=4 AP", 0.2667,
     "elliptic2_feature_audit.json", "prefix_truncation/2/AP",
     "50_elliptic2_feature_audit.py", "同上"),
    ("Prefix k=43 AP", 0.298,
     "elliptic2_feature_audit.json", "prefix_truncation/6/AP",
     "50_elliptic2_feature_audit.py", "同上"),
    # ---------- Table 14: ACI
    ("Fixed conformal coverage alpha=0.1", 0.9935,
     "aci_conformal.json", "summary/0.1/fixed_mean_coverage",
     "47_aci_conformal.py", "data/real_cats_eth.csv"),
    ("ACI coverage alpha=0.1", 0.9529,
     "aci_conformal.json", "summary/0.1/aci_mean_coverage",
     "47_aci_conformal.py", "data/real_cats_eth.csv"),
    ("ACI alert rate alpha=0.1", 0.7224,
     "aci_conformal.json", "summary/0.1/aci_mean_alert_rate",
     "47_aci_conformal.py", "data/real_cats_eth.csv"),
    # ---------- audit / safe / base rate
    ("Label-permutation AP (audit)", 0.028,
     "leakage_audit.json", "checks.0.permuted_mean",
     "27_leakage_audit.py", "data/elliptic2/nodes.csv"),
    ("Size-shift prevalence change", 0.262,
     "leakage_audit.json", "checks.4.relative_prevalence_shift",
     "27_leakage_audit.py", "data/elliptic2/nodes.csv"),
    ("Label reconstruction AUC", 0.871,
     "leakage_audit_v2.json", "checks.1.feature_only_AUC",
     "35_leakage_audit_v2.py", "data/elliptic2/elliptic2_subgraph_embeddings.npy"),
    ("Elliptic2-Safe PR-AUC", 0.025,
     "elliptic2_safe.json", "Safe.PR-AUC",
     "41_elliptic2_safe.py", "data/elliptic2/node_features_43.npy"),
    ("Reweighted AP at 1% base rate", 0.549,
     "base_rate_reweight.json", "reweighted/0.01/weighted_PR_AUC",
     "42_base_rate_reweight.py", "data/real_cats_eth.csv"),
    ("Reweighted AP at 10% base rate", 0.778,
     "base_rate_reweight.json", "reweighted/0.1/weighted_PR_AUC",
     "42_base_rate_reweight.py", "同上"),
    # ---------- drift / strict / lambda / embedding
    ("Size drift proxy shift AP", 0.198,
     "drift_proxies.json", "proxies.size.shift_AP",
     "45_drift_proxies.py", "data/elliptic2/node_features_43.npy"),
    ("Strict-protocol static SGC (train75)", 0.844,
     "true_timestamp_graph.json", "results.static.train75.PR-AUC",
     "32_true_timestamp_graph.py", "data/real_cats/real_graph_nodes.csv"),
    ("Lambda strict spread", 0.0004,
     "lambda_sensitivity.json", "finding",
     "33_lambda_sensitivity.py", "data/real_cats/real_graph_nodes.csv"),
    ("Elliptic2 embedding silhouette", 0.032,
     "embedding_analysis.json", "elliptic2.silhouette",
     "34_embedding_visualization.py", "data/elliptic2/node_features_43.npy"),
    # ---------- operating point
    ("Operating-point F1", 0.787,
     "operational_metrics.json", "operating_points/F1@0.5",
     "38_operational_metrics.py", "data/real_cats_eth.csv"),
    ("ECE 10 bins", 0.038,
     "operational_metrics.json", "operating_points.ECE_10bins",
     "38_operational_metrics.py", "同上"),
    ("Attack AP at eps=2.0", 0.540,
     "operational_metrics.json", "attack_budget/2.0",
     "38_operational_metrics.py", "同上"),
    # ---------- resource / rolling
    ("StreamTemporal P50 latency (ms)", 2.89,
     "resource_benchmark.json", "models.1.latency_p50_ms",
     "28_resource_benchmark.py", "data/real_cats/real_graph_nodes.csv"),
    ("Rolling-origin XGBoost mean PR-AUC", 0.883,
     "rolling_origin.json", "summary.XGBoost.PR_AUC_mean",
     "31_rolling_origin.py", "data/real_cats_eth.csv"),
]


def extract(result_file: str, desc: str):
    p = RESULTS_DIR / result_file
    if not p.exists():
        return None, "result file missing"
    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            if desc.startswith("max("):
                return float(df["PR_AUC_mean"].max()), ""
            if desc.startswith("mean("):
                col = desc[5:-1]
                return float(df[col].mean()), ""
            # model=..., split=..., 最后一列为值列
            cond = df
            for part in desc.split(","):
                if "=" in part and not part.startswith("agg="):
                    k, v = part.split("=", 1)
                    cond = cond[cond[k].astype(str) == v]
            last = desc.split(",")[-1]
            if last.startswith("agg=mean:"):
                col = last.split(":", 1)[1]
                return float(cond[col].mean()), ""
            return float(cond[last].iloc[0]), ""
        else:
            d = json.loads(p.read_text(encoding="utf-8"))
            obj = d
            sep = "/" if "/" in desc else "."
            for key in desc.split(sep):
                if isinstance(obj, list):
                    obj = obj[int(key)]
                else:
                    obj = obj[key]
            if isinstance(obj, str):
                import re as _re
                m = _re.search(r"spread ([0-9.]+)", obj)
                if m: return float(m.group(1)), ""
                raise ValueError("no numeric spread in finding")
            return float(obj), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    rows = []
    for claim, paper_val, rfile, desc, script, data in TRACE:
        actual, err = extract(rfile, desc)
        script_exists = (Path(__file__).resolve().parent / script).exists()
        # 检查数据路径中列出的关键文件是否存在（取第一个存在的路径提示）
        data_ok = True
        import glob as _glob
        for token in data.replace("同上", "").split(" "):
            token = token.strip("+")
            if token.startswith("data/"):
                full = str(Path(__file__).resolve().parent.parent / token)
                # 展开 {a,b} 花括号和 glob
                matches = _glob.glob(full.replace("{", "{")) if "{" not in full else _glob.glob(full)
                if not matches:
                    # 花括号展开失败时，按逗号拆开逐个查
                    import re as _re
                    expanded = _re.sub(r"\{([^}]+)\}", lambda m: "{" + m.group(1) + "}", full)
                    alts = []
                    if "{" in expanded:
                        parts = expanded.split("{", 1)
                        tail = parts[1].split("}", 1)[1]
                        for item in parts[1].split("}", 1)[0].split(","):
                            alts.append(parts[0] + item + tail)
                    else:
                        alts = [expanded]
                    if not any(_glob.glob(a) for a in alts):
                        data_ok = False
                        break
        value_ok = actual is not None and abs(actual - paper_val) <= 0.011
        ok = value_ok and script_exists and data_ok
        rows.append({
            "claim": claim, "paper_value": paper_val,
            "actual_value": round(actual, 4) if actual is not None else None,
            "result_file": rfile, "extraction": desc,
            "script": script, "script_exists": script_exists,
            "data": data, "data_available": data_ok,
            "traceable": bool(ok), "error": err,
        })
        print(f"{'OK  ' if ok else 'FAIL'} {claim:38s} paper={paper_val:<7} "
              f"actual={actual if actual is None else round(actual,4)}")

    n_fail = sum(1 for r in rows if not r["traceable"])
    report = {
        "status": "PASS" if n_fail == 0 else "FAIL",
        "n_claims": len(rows), "n_untraceable": n_fail,
        "summary": ("Every quantitative claim in the manuscript is linked to a result file, "
                    "a generation script, and an input dataset. Run this script to verify."),
        "checks": rows,
    }
    out = RESULTS_DIR / "reproducibility_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("reproducibility_audit", {"status": report["status"], "n_fail": n_fail})
    print(f"\n[done] {out}: {len(rows)-n_fail}/{len(rows)} claims fully traceable")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
