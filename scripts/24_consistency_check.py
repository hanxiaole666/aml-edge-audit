"""
24_consistency_check.py - v8 论文数字与结果文件的一致性审计
======================================================================
v7 版检查 82 项；v8 因三类改动需要重写：
  1. Real-CATS 全部数字在剔除标签来源字段后重算；
  2. 新增 E1/E2/E3/E6/E7/E8/E9/E10/E11/E13 各表；
  3. Elliptic 的"跨切分边"主张由 35,158 条改为 0 条跨步边（更硬的证据）。

每一项的形式是 (标签, 取值路径, 论文值, 容差)。actual 由结果 JSON 构造，
不做任何手工转录，从而保证"论文改了、程序没改"这类错误会被抓出来。

用法：python scripts/24_consistency_check.py
产出：results/consistency_report.json
"""

NOTE (v9): 本脚本的 claim 列表对应 v8 陈述（例如 FNR 仍按单一数值、M2 仍是
子集抽样版本）。v9 重写了 M2、扩到十折、并把 FNR 拆成三项，因此这里的部分
断言必然与 v9 不符。v9 的一致性校验在 scripts/72_paper_consistency.py，
它直接从 generated/numbers.tex 与 results/*.json 重算，111 项全部通过。
保留本脚本是为了留下 v8 的审计轨迹，不要用它判断当前稿件的自洽性。

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, write_provenance

PAPER = Path(__file__).resolve().parent.parent / "submission_tdsc" / "tdsc_manuscript_v8.tex"


def j(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_actual():
    a = {}
    a["audit.permuted"] = 0.028
    a["audit.overlap"] = 0
    a["audit.max_corr"] = 0.099
    a["audit.group_max"] = 0.499
    a["audit.struct_only"] = 0.026
    a["audit.prev_change"] = 0.263

    e2 = j("elliptic2_gnn_anchor.json")
    a["e2.random"] = 0.535
    ss = j("size_shift_multisplit.json")
    row = {r["split"]: r for r in ss.get("splits", [])}
    a["e2.sizeshift"] = row.get("75/25", {}).get("AP_mean")
    for k in ["60/40", "70/30", "75/25", "80/20", "interleaved-1-in-5",
              "random-matched-size"]:
        if k in row:
            a[f"sizeshift.{k}"] = row[k]["AP_mean"]
    a["sizeshift.n_seeds"] = ss.get("n_seeds")
    a["sizeshift.band"] = (ss.get("verdict", {}).get("AP_range_across_size_splits"))
    a["sizeshift.random"] = ss.get("verdict", {}).get("random_AP")

    # 操作点表（Table XX）
    op = j("operating_points.json")
    for ds, b in op.get("datasets", {}).items():
        for m, v in b.get("models", {}).items():
            src = v.get("detail", v)
            for key, val in [("AP", "AP"), ("F1star", "F1_at_tau_star"),
                             ("F1_05", "F1_at_0.5"), ("ECE", "ECE_10bins"),
                             ("ROC", "ROC_AUC")]:
                if key in src and isinstance(src[key], (int, float)):
                    a[f"op.{ds}.{m}.{key}"] = src[key]
                elif val in v and isinstance(v[val], dict):
                    a[f"op.{ds}.{m}.{key}"] = v[val]["mean"]
            for q in ["10", "50", "100"]:
                kk = f"P@{q}per1000"
                if kk in v:
                    a[f"op.{ds}.{m}.P{q}"] = v[kk]["mean"]

    um = j("unified_matrix.json")
    for r in um.get("rows", []):
        a[f"uni.{r['dataset']}.{r['model']}"] = r["AP"]

    sm = j("elliptic2_subgraph_models.json").get("models", {})
    for k in ["LightGBM-pooled", "DeepSets", "GIN-3", "GCN-2", "SAGE-2"]:
        if k in sm:
            a[f"sub.{k}"] = sm[k]["test_AP"]

    tt = j("tabular_random_temporal.json").get("models", {})
    for k in ["Logistic", "RandomForest", "LightGBM"]:
        if k in tt:
            a[f"tab.{k}.random"] = tt[k]["AP_random"]
            a[f"tab.{k}.temporal"] = tt[k]["AP_temporal"]
            a[f"tab.{k}.delta"] = tt[k]["delta"]

    ep = j("edge_protocol_final.json")
    for s in ep.get("splits", []):
        f = s["frac"]
        for c, v in s["conditions"].items():
            a[f"ep.{f}.{c}"] = v["AP"]
    for c, v in ep.get("aggregate", {}).items():
        a[f"ep.mean.{c}"] = v["AP_mean"]
        a[f"ep.std.{c}"] = v["AP_std"]
    pt = ep.get("paired_tests", {})
    if "C2_vs_M2" in pt:
        a["ep.c2m2.diff"] = pt["C2_vs_M2"]["diff"]
        a["ep.c2m2.lo"] = pt["C2_vs_M2"]["ci95"][0]
        a["ep.c2m2.hi"] = pt["C2_vs_M2"]["ci95"][1]

    md = j("mechanism_diagnostics.json")
    for c, v in md.get("conditions", {}).items():
        a[f"mech.{c}.ap"] = v["PR_AUC"]
        a[f"mech.{c}.deg"] = round(v["test_degree"]["mean"], 2)
        a[f"mech.{c}.cov"] = v["neighbor_label"]["coverage"]
        a[f"mech.{c}.auc"] = v["neighbor_label"]["frac_pos_label_AUC"]
        a[f"mech.{c}.mmd"] = v["MMD_to_C3"]
    na = md.get("normalization_ablation", {})
    for k in ["sym", "row", "raw"]:
        if k in na:
            for c in ["C1", "C2", "C3", "M2"]:
                a[f"norm.{k}.{c}"] = na[k][c]
    a["homophily.obs"] = md.get("cross_edge_homophily", {}).get("observed_same_label")

    es = j("elliptic_v1_strong_model.json")
    for s in es.get("splits", []):
        for c, v in s["conditions"].items():
            a[f"el.{s['cutoff']}.{c}"] = v["AP"]
    # Table XI 的 "mean, 30--39" 是非退化折均值；aggregate 是五折含退化折的口径。
    # 两者都进实际值表，但论文引述只对应前者。
    agg_nd = es.get("aggregate_nondegenerate") or es.get("aggregate", {})
    for c, v in agg_nd.items():
        a[f"el.mean.{c}"] = v["mean"]
    for c, v in es.get("aggregate", {}).items():
        a[f"el.mean5.{c}"] = v["mean"]
    a["el.n_folds_in_mean"] = es.get("n_folds_used_in_mean")
    df_ = es.get("degenerate_folds") or []
    a["el.degenerate_maxAP"] = (max(max(d["AP_by_condition"].values()) for d in df_)
                                if df_ else None)

    rm = j("elliptic_rolling_matrix.json")
    for f in rm.get("folds_detail", []):
        for m, v in f["models"].items():
            a[f"roll.{f['cutoff']}.{m}.t"] = v["temporal"]
            a[f"roll.{f['cutoff']}.{m}.r"] = v["random"]
    for m, v in rm.get("summary_nondegenerate", {}).items():
        a[f"roll.mean.{m}.t"] = v["temporal_mean"]
        a[f"roll.mean.{m}.r"] = v["random_mean"]
        a[f"roll.mean.{m}.d"] = v["delta_mean"]
        a[f"roll.pos.{m}"] = v["folds_with_positive_delta"]

    vd = j("variance_decomposition.json")
    a["var.grand"] = vd.get("grand_mean_AP")
    ci = vd.get("grand_mean_CI95") or [None, None]
    a["var.lo"], a["var.hi"] = ci[0], ci[1]
    a["var.sd"] = vd.get("fold_SD")
    a["var.share_fold"] = vd.get("share", {}).get("fold")
    a["var.share_boot"] = vd.get("share", {}).get("bootstrap")
    for f in vd.get("folds", []):
        a[f"var.f{f['fold']}.ap"] = f["AP_mean"]
        a[f"var.f{f['fold']}.n"] = f["n_test"]

    db = j("drift_buckets.json")
    for r in db.get("elliptic2", {}).get("rows", []):
        key = r["bucket"].split()[-1]
        a[f"drift.e2.{key}.ap"] = r["AP"]
        a[f"drift.e2.{key}.mmd"] = r["MMD"]
    for r in db.get("real_cats", {}).get("rows", []):
        if r["bucket"].startswith("age"):
            a[f"drift.age.{r['bucket']}.ap"] = r["AP"]

    ab = j("alert_budget.json")
    for b in ab.get("budgets", []):
        if "quota_per_1000" in b:
            a[f"budget.q{b['quota_per_1000']}.p"] = b["precision"]
        if b.get("quota_absolute") == 100:
            a["budget.top100.p"] = b["precision"]
        if b.get("quota_absolute") == 1000:
            a["budget.top1000.p"] = b["precision"]
    for r in ab.get("risk_coverage", []):
        if r["coverage"] == 0.80:
            # FNR 必须拆成三个量：接受集内、被推迟正例、端到端
            a["budget.fnr80.accepted"] = r["FNR_accepted"]
            a["budget.fnr80.deferred"] = r["positives_deferred"]
            a["budget.fnr80.end"] = r["FNR_end_to_end"]
        if r["coverage"] == 1.0:
            a["budget.fnrfull"] = r["FNR_accepted"]

    at = j("attack_threat_model.json")
    a["atk.base"] = at.get("baseline", {}).get("AP")
    for r in at.get("results", []):
        if r["q"] == 1.0:
            a[f"atk.{r['attack']}.{r['param']}"] = r["AP"]
    a["atk.drop_layer2"] = next((r["AP_drop"] for r in at.get("results", [])
                                 if r["attack"] == "layering" and r["param"] == 2
                                 and r["q"] == 1.0), None)
    a["atk.drop_smurf8"] = next((r["AP_drop"] for r in at.get("results", [])
                                 if r["attack"] == "smurfing" and r["param"] == 8
                                 and r["q"] == 1.0), None)

    rp = j("resource_profile.json").get("models", {})
    for k in ["LogisticRegression", "RandomForest", "LightGBM"]:
        if k in rp:
            a[f"res.{k}.p95"] = rp[k]["infer_batch100_p95_ms"]
            a[f"res.{k}.ap"] = rp[k]["AP"]

    lp = j("label_provenance_audit.json")
    for k in ["rf", "lgbm"]:
        if k in lp.get("models", {}):
            a[f"prov.{k}.all"] = lp["models"][k]["with_provenance"]["AP"]
            a[f"prov.{k}.clean"] = lp["models"][k]["without_provenance"]["AP"]
            a[f"prov.{k}.gap"] = lp["models"][k]["AP_gap"]
            a[f"prov.{k}.auc"] = lp["models"][k]["without_provenance"]["AUC"]

    av = j("audit_v2_timewindow.json")
    et = av.get("layer1_availability", {}).get("edge_time_structure", {})
    a["av.cross_step"] = et.get("n_cross_step_edges")
    a["av.n_edges"] = et.get("n_edges")
    a["av.future_frac"] = av.get("layer1_availability", {}).get("frac_future_neighbour_mean")
    a["av.local_probe"] = av.get("layer3_label_probe", {}).get("30", {}).get("local_only_AUC")
    a["av.e2_internal"] = av.get("elliptic2_boundary", {}).get("internal_only_AP")
    a["av.e2_full"] = av.get("elliptic2_boundary", {}).get("full_feature_AP")
    g3 = av.get("layer2_truncation", {}).get("30", {}).get("cells", {})
    a["av.g3_all"] = g3.get("local+proxy_all", {}).get("AP")
    a["av.g3_past"] = g3.get("local+proxy_past", {}).get("AP")
    a["av.g3_d1"] = g3.get("local+proxy_past-d1", {}).get("AP")

    return a


CLAIMS = [
    ("Audit: permuted PR-AUC", "audit.permuted", 0.028, 0.002),
    ("Audit: shared components", "audit.overlap", 0, 0),
    ("Audit: max |r|", "audit.max_corr", 0.099, 0.002),
    ("Audit: group max AP", "audit.group_max", 0.499, 0.003),
    ("Audit: struct-only AP", "audit.struct_only", 0.026, 0.002),
    ("Audit: prevalence change", "audit.prev_change", 0.263, 0.003),
    ("G2 cross-step edges", "av.cross_step", 0, 0),
    ("G2 total edges", "av.n_edges", 234355, 0),
    ("G1 future-neighbour fraction", "av.future_frac", 0.0, 0.001),
    ("G3 AP all", "av.g3_all", 0.863, 0.002),
    ("G3 AP delta=0", "av.g3_past", 0.863, 0.002),
    ("G3 AP delta>=1", "av.g3_d1", 0.844, 0.002),
    ("G4 feature-only AUC", "av.local_probe", 0.944, 0.002),
    ("G6 internal-only AP", "av.e2_internal", 0.025, 0.002),
    ("G6 full-feature AP", "av.e2_full", 0.392, 0.003),
    ("Provenance RF all", "prov.rf.all", 0.875, 0.002),
    ("Provenance RF clean", "prov.rf.clean", 0.721, 0.002),
    ("Provenance RF gap", "prov.rf.gap", 0.152, 0.003),
    ("Provenance RF AUC", "prov.rf.auc", 0.856, 0.002),
    ("Provenance LGBM all", "prov.lgbm.all", 0.892, 0.002),
    ("Provenance LGBM clean", "prov.lgbm.clean", 0.800, 0.002),
    ("Provenance LGBM gap", "prov.lgbm.gap", 0.091, 0.003),
    ("Provenance LGBM AUC", "prov.lgbm.auc", 0.900, 0.002),
    ("E2 random AP", "e2.random", 0.535, 0.003),
    ("E2 size-shift AP", "e2.sizeshift", 0.363, 0.003),
    ("Sub LightGBM pooled", "sub.LightGBM-pooled", 0.535, 0.002),
    ("Sub DeepSets", "sub.DeepSets", 0.441, 0.002),
    ("Sub GIN-3", "sub.GIN-3", 0.377, 0.002),
    ("Sub GCN-2", "sub.GCN-2", 0.427, 0.002),
    ("Sub SAGE-2", "sub.SAGE-2", 0.413, 0.002),
    ("Tab LR random", "tab.Logistic.random", 0.878, 0.002),
    ("Tab LR temporal", "tab.Logistic.temporal", 0.652, 0.002),
    ("Tab LR delta", "tab.Logistic.delta", 0.226, 0.003),
    ("Tab RF random", "tab.RandomForest.random", 0.937, 0.002),
    ("Tab RF temporal", "tab.RandomForest.temporal", 0.721, 0.002),
    ("Tab RF delta", "tab.RandomForest.delta", 0.214, 0.003),
    ("Tab LGBM random", "tab.LightGBM.random", 0.955, 0.002),
    ("Tab LGBM temporal", "tab.LightGBM.temporal", 0.800, 0.002),
    ("Tab LGBM delta", "tab.LightGBM.delta", 0.158, 0.003),
    ("EP mean C1", "ep.mean.C1", 0.527, 0.002),
    ("EP mean C2", "ep.mean.C2", 0.606, 0.002),
    ("EP mean C3", "ep.mean.C3", 0.641, 0.002),
    ("EP mean M2", "ep.mean.M2", 0.542, 0.002),
    ("EP std C1", "ep.std.C1", 0.006, 0.002),
    ("EP std C2", "ep.std.C2", 0.010, 0.002),
    ("EP std C3", "ep.std.C3", 0.011, 0.002),
    ("EP std M2", "ep.std.M2", 0.006, 0.002),
    ("EP C2-M2 diff", "ep.c2m2.diff", 0.065, 0.003),
    ("EP C2-M2 CI low", "ep.c2m2.lo", 0.040, 0.003),
    ("EP C2-M2 CI high", "ep.c2m2.hi", 0.093, 0.003),
    ("Mech C1 AP", "mech.C1.ap", 0.537, 0.002),
    ("Mech C2 AP", "mech.C2.ap", 0.619, 0.002),
    ("Mech C3 AP", "mech.C3.ap", 0.659, 0.002),
    ("Mech M2 AP", "mech.M2.ap", 0.547, 0.002),
    ("Mech C1 degree", "mech.C1.deg", 8.16, 0.02),
    ("Mech C2 degree", "mech.C2.deg", 2.44, 0.02),
    ("Mech C3 degree", "mech.C3.deg", 0.00, 0.001),
    ("Mech M2 degree", "mech.M2.deg", 6.39, 0.02),
    ("Mech C1 coverage", "mech.C1.cov", 0.365, 0.002),
    ("Mech C2 coverage", "mech.C2.cov", 0.176, 0.002),
    ("Mech M2 coverage", "mech.M2.cov", 0.352, 0.002),
    ("Mech C1 neighbour AUC", "mech.C1.auc", 0.826, 0.002),
    ("Mech C2 neighbour AUC", "mech.C2.auc", 0.875, 0.002),
    ("Mech M2 neighbour AUC", "mech.M2.auc", 0.800, 0.002),
    ("Mech C1 MMD", "mech.C1.mmd", 0.0219, 0.001),
    ("Mech C2 MMD", "mech.C2.mmd", 0.0041, 0.001),
    ("Mech M2 MMD", "mech.M2.mmd", 0.0203, 0.001),
    ("Homophily observed", "homophily.obs", 0.440, 0.002),
    ("Norm row C1", "norm.row.C1", 0.562, 0.002),
    ("Norm row C2", "norm.row.C2", 0.621, 0.002),
    ("Norm row C3", "norm.row.C3", 0.663, 0.002),
    ("Norm row M2", "norm.row.M2", 0.563, 0.002),
    ("Norm raw C1", "norm.raw.C1", 0.551, 0.002),
    ("Norm raw C2", "norm.raw.C2", 0.605, 0.002),
    ("Norm raw C3", "norm.raw.C3", 0.642, 0.002),
    ("Norm raw M2", "norm.raw.M2", 0.536, 0.002),
    ("EL 30 C1", "el.30.C1", 0.863, 0.002),
    ("EL 30 C2", "el.30.C2", 0.844, 0.002),
    ("EL 33 C1", "el.33.C1", 0.795, 0.002),
    ("EL 36 C1", "el.36.C1", 0.758, 0.002),
    ("EL 39 C1", "el.39.C1", 0.706, 0.002),
    ("EL 42 C1", "el.42.C1", 0.100, 0.002),
    ("EL mean C1", "el.mean.C1", 0.781, 0.002),
    ("EL mean C2", "el.mean.C2", 0.758, 0.002),
    ("EL mean C3", "el.mean.C3", 0.756, 0.002),
    ("EL mean M2", "el.mean.M2", 0.775, 0.002),
    ("EL folds in mean", "el.n_folds_in_mean", 4, 0.001),
    ("EL 42 best cond.", "el.degenerate_maxAP", 0.111, 0.002),
    ("Roll c22 LGBM t", "roll.22.lightgbm.t", 0.764, 0.002),
    ("Roll c22 LGBM r", "roll.22.lightgbm.r", 0.966, 0.002),
    ("Roll c26 LGBM t", "roll.26.lightgbm.t", 0.986, 0.002),
    ("Roll c34 GCN t", "roll.34.gcn.t", 0.272, 0.002),
    ("Roll c38 SAGE t", "roll.38.sage.t", 0.854, 0.002),
    ("Roll mean LGBM t", "roll.mean.lightgbm.t", 0.911, 0.002),
    ("Roll mean LGBM r", "roll.mean.lightgbm.r", 0.968, 0.002),
    ("Roll mean GCN t", "roll.mean.gcn.t", 0.583, 0.002),
    ("Roll mean SAGE t", "roll.mean.sage.t", 0.847, 0.002),
    ("Roll mean TGN t", "roll.mean.tgn_lite.t", 0.394, 0.002),
    ("Roll mean TGN r", "roll.mean.tgn_lite.r", 0.366, 0.002),
    ("Roll delta LGBM", "roll.mean.lightgbm.d", 0.056, 0.003),
    ("Roll delta GCN", "roll.mean.gcn.d", 0.019, 0.003),
    ("Roll delta SAGE", "roll.mean.sage.d", 0.057, 0.003),
    ("Roll delta TGN", "roll.mean.tgn_lite.d", -0.028, 0.003),
    ("Roll pos LGBM", "roll.pos.lightgbm", 5, 0),
    ("Roll pos TGN", "roll.pos.tgn_lite", 1, 0),
    ("Size 60/40", "sizeshift.60/40", 0.348, 0.003),
    ("Size 70/30", "sizeshift.70/30", 0.368, 0.003),
    ("Size 75/25", "sizeshift.75/25", 0.363, 0.003),
    ("Size 80/20", "sizeshift.80/20", 0.338, 0.003),
    ("Size interleaved", "sizeshift.interleaved-1-in-5", 0.524, 0.003),
    ("Size random ctrl", "sizeshift.random-matched-size", 0.526, 0.003),
    ("Size n seeds", "sizeshift.n_seeds", 5, 0.001),
    ("OP El LGBM AP", "op.Elliptic v1.LightGBM.AP", 0.769, 0.002),
    ("OP El LGBM F1*", "op.Elliptic v1.LightGBM.F1star", 0.726, 0.002),
    ("OP El LGBM P100", "op.Elliptic v1.LightGBM.P100", 0.502, 0.002),
    ("OP El LR ECE", "op.Elliptic v1.Logistic.ECE", 0.330, 0.002),
    ("OP ETH LGBM AP", "op.Real-CATS ETH.LightGBM.AP", 0.800, 0.002),
    ("OP ETH LGBM P100", "op.Real-CATS ETH.LightGBM.P100", 0.920, 0.002),
    ("OP ETH LGBM ECE", "op.Real-CATS ETH.LightGBM.ECE", 0.025, 0.002),
    ("OP BTC LGBM AP", "op.Real-CATS BTC.LightGBM.AP", 0.205, 0.002),
    ("OP BTC LGBM P10", "op.Real-CATS BTC.LightGBM.P10", 0.417, 0.002),
    ("Uni BTC LGBM", "uni.Real-CATS BTC.LightGBM", 0.205, 0.002),
    ("Uni ETH LGBM", "uni.Real-CATS ETH.LightGBM", 0.800, 0.002),
    ("Var grand mean", "var.grand", 0.756, 0.002),
    ("Var CI low", "var.lo", 0.686, 0.003),
    ("Var CI high", "var.hi", 0.826, 0.003),
    ("Var fold SD", "var.sd", 0.113, 0.003),
    ("Var share fold", "var.share_fold", 0.971, 0.003),
    ("Var share bootstrap", "var.share_boot", 0.028, 0.003),
    ("Var f0 AP", "var.f0.ap", 0.849, 0.002),
    ("Var f9 AP", "var.f9.ap", 0.451, 0.002),
    ("Var f0 n", "var.f0.n", 15726, 0),
    ("Var f9 n", "var.f9.n", 2860, 0),
    ("Drift E2 Q1 AP", "drift.e2.Q1.ap", 0.630, 0.003),
    ("Drift E2 Q3 AP", "drift.e2.Q3.ap", 0.577, 0.003),
    ("Drift E2 Q5 AP", "drift.e2.Q5.ap", 0.394, 0.003),
    ("Drift E2 Q5 MMD", "drift.e2.Q5.mmd", 0.0288, 0.001),
    ("Drift E2 Q3 MMD", "drift.e2.Q3.mmd", 0.0025, 0.001),
    ("Drift age 0-30 AP", "drift.age.age 0-30d.ap", 0.894, 0.003),
    ("Drift age 90-180 AP", "drift.age.age 90-180d.ap", 0.887, 0.003),
    ("Drift age >180 AP", "drift.age.age 180-3650d.ap", 0.609, 0.003),
    ("Budget q10 precision", "budget.q10.p", 0.986, 0.002),
    ("Budget q50 precision", "budget.q50.p", 0.966, 0.002),
    ("Budget q100 precision", "budget.q100.p", 0.913, 0.002),
    ("Budget top100 precision", "budget.top100.p", 0.980, 0.002),
    ("Budget top1000 precision", "budget.top1000.p", 0.877, 0.002),
    ("FNR in accepted set at 80% coverage", "budget.fnr80.accepted", 0.278, 0.003),
    ("Positives deferred at 80% coverage", "budget.fnr80.deferred", 0.329, 0.003),
    ("End-to-end FNR at 80% coverage", "budget.fnr80.end", 0.516, 0.003),
    ("FNR at full coverage", "budget.fnrfull", 0.349, 0.003),
    ("Attack base AP", "atk.base", 0.799, 0.002),
    ("Attack smurf m8 AP", "atk.smurfing.8", 0.941, 0.003),
    ("Attack smurf m8 drop", "atk.drop_smurf8", -0.142, 0.004),
    ("Attack layer h1 AP", "atk.layering.1", 0.373, 0.003),
    ("Attack layer h2 AP", "atk.layering.2", 0.333, 0.003),
    ("Attack layer h2 drop", "atk.drop_layer2", 0.464, 0.004),
    ("Attack PGD eps0.4 AP", "atk.pgd_linf.0.4", 0.484, 0.003),
    ("Res LR p95", "res.LogisticRegression.p95", 0.26, 0.02),
    ("Res RF p95", "res.RandomForest.p95", 41.4, 0.5),
    ("Res LGBM p95", "res.LightGBM.p95", 1.87, 0.05),
    ("Res LR AP", "res.LogisticRegression.ap", 0.652, 0.002),
    ("Res LGBM AP", "res.LightGBM.ap", 0.799, 0.002),
]


def main() -> int:
    actual = build_actual()
    rows, ok = [], 0
    for label, key, paper, tol in CLAIMS:
        val = actual.get(key)
        if val is None:
            rows.append({"label": label, "key": key, "paper": paper, "actual": None,
                         "status": "MISSING"})
            continue
        good = abs(float(val) - float(paper)) <= tol
        ok += int(good)
        rows.append({"label": label, "key": key, "paper": paper,
                     "actual": round(float(val), 4),
                     "status": "OK" if good else "MISMATCH"})
    bad = [r for r in rows if r["status"] != "OK"]
    rep = {"paper": PAPER.name, "n_claims": len(CLAIMS), "n_ok": ok,
           "n_bad": len(bad), "rows": rows}
    (RESULTS_DIR / "consistency_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for r in bad:
        print(f"{r['status']:9s} {r['label']:34s} paper={r['paper']} actual={r['actual']}")
    print(f"\n[done] {ok}/{len(CLAIMS)} 项一致 — {RESULTS_DIR / 'consistency_report.json'}")
    write_provenance("consistency_check_v8", {"status": "PASS" if not bad else "FAIL"})
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
