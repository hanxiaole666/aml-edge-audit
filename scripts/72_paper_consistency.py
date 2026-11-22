"""
72_paper_consistency.py - v9 论文数字一致性校验
================================================
校验三层：
  A. 宏值层：generated/numbers.tex 里每个宏的值，是否等于从 results/*.json
     独立重算出来的值（容差按量级取 1e-3 或 1e-4）。
  B. 结构层：本轮结论所依赖的方向性断言（M2 是否在每个切分最低、十折符号
     是否全同、注入是否单调、LOGO 的 R^2 是否全负、缓解后 AP_drop 是否 <=0）。
  C. 文本层：正文是否残留旧版数字或旧表述；正文引用的宏是否都已定义。

A 层防"抄错"，B 层防"结论与数据不符"，C 层防"改了数据没改正文"。

用法：python scripts/72_paper_consistency.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SUB = ROOT / "submission_tdsc"
NUM = SUB / "generated" / "numbers.tex"
TEX = SUB / "tdsc_manuscript_v15.tex"

TOL = 1e-3          # 三位小数宏
TOL4 = 5e-5         # 四位小数宏（MMD、ΔAP）

fails: list[str] = []
checks = 0


def j(name: str) -> dict:
    with open(RES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def macro_table() -> dict[str, str]:
    text = NUM.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))


def num(s: str) -> float | None:
    s = s.replace("$", "").replace("\\pm", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def check(macros: dict[str, str], name: str, expect: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    if name not in macros:
        fails.append(f"MISSING MACRO {name}")
        return
    got = num(macros[name])
    if got is None:
        fails.append(f"NON-NUMERIC {name} = {macros[name]!r}")
        return
    if abs(got - expect) > tol:
        fails.append(f"MISMATCH {name}: tex={got} data={expect:.6f}")


def check_eq(macros: dict[str, str], name: str, expect: str) -> None:
    global checks
    checks += 1
    if name not in macros:
        fails.append(f"MISSING MACRO {name}")
    elif macros[name].strip() != expect:
        fails.append(f"MISMATCH {name}: tex={macros[name]!r} data={expect!r}")


def main() -> int:
    m = macro_table()

    # ---------------------------------------------------------------- A. 宏值
    ep = j("edge_protocol_final")
    for k, tag in (("C1", "cone"), ("C2", "ctwo"), ("C3", "cthree"), ("M2", "mtwo")):
        check(m, f"RCC{tag}", ep["aggregate"][k]["AP_mean"])
        check(m, f"RCC{tag}sd", ep["aggregate"][k]["AP_std"])
    check(m, "RCCthreeGain", ep["aggregate"]["C3"]["AP_mean"] - ep["aggregate"]["C1"]["AP_mean"])
    check(m, "RCCmTwoGap", ep["aggregate"]["C1"]["AP_mean"] - ep["aggregate"]["M2"]["AP_mean"])
    lo, hi = ep["paired_tests"]["C2_vs_C3"]["ci95"]
    check_eq(m, "RCCpairCtwoCthreeCI", f"[{lo:.3f}, {hi:.3f}]")

    md = j("mechanism_diagnostics")["conditions"]
    for k, tag in (("C1", "cone"), ("C2", "ctwo"), ("C3", "cthree"), ("M2", "mtwo")):
        check(m, f"MECH{tag}ap", md[k]["PR_AUC"])
        check(m, f"MECH{tag}deg", md[k]["test_degree"]["mean"], 6e-3)
        check(m, f"MECH{tag}cov", 100 * md[k]["neighbor_label"]["coverage"], 0.06)
    check(m, "MECHmtwocov", 100 * md["M2"]["neighbor_label"]["coverage"], 0.06)

    hom = j("cross_edge_homophily")
    r75 = next(r for r in hom["real_cats"] if r["split"] == "frac=0.75")
    check(m, "HOMobs", r75["cross_edges"]["observed_same"])
    check(m, "HOMexp", r75["cross_edges"]["expected_independent"])
    check(m, "HOMexcess", r75["cross_edges"]["excess"])
    check(m, "HOMexcessMean",
          sum(r["cross_edges"]["excess"] for r in hom["real_cats"]) / len(hom["real_cats"]))

    el = j("elliptic_v1_strong_model")["aggregate_nondegenerate"]
    for k, tag in (("C1", "one"), ("C2", "two"), ("C3", "three"), ("M2", "mtwo")):
        check(m, f"ELC{tag}", el[k]["mean"])
    check(m, "ELConeMinusCthree", el["C1"]["mean"] - el["C3"]["mean"])

    ro = j("elliptic_rolling_matrix")
    sd = ro["summary"]
    check(m, "ROLgbTemporal", sd["lightgbm"]["temporal_mean"])
    check(m, "ROLgbRandom", sd["lightgbm"]["random_mean"])
    check(m, "ROLgbDelta", sd["lightgbm"]["delta_mean"])
    check(m, "ROTgnDelta", sd["tgn_lite"]["delta_mean"])
    check_eq(m, "ROnFolds", str(len(ro["folds_detail"])))
    check_eq(m, "ROLgbPos", str(sd["lightgbm"]["delta_positive_folds"]))

    sub = j("elliptic2_subgraph_models")
    ss = j("size_shift_multisplit")
    check(m, "ETrand", sub["models"]["LightGBM-pooled"]["test_AP"])
    s75 = next(r for r in ss["splits"] if r["split"] == "75/25")
    check(m, "ETsize", s75["AP_mean"])
    check(m, "ETdrop", 100 * (sub["models"]["LightGBM-pooled"]["test_AP"] - s75["AP_mean"])
          / sub["models"]["LightGBM-pooled"]["test_AP"], 0.06)
    check(m, "SSlo", min(r["AP_mean"] for r in ss["splits"]
                         if r["split"] in ("60/40", "70/30", "75/25", "80/20")))
    check(m, "SShi", max(r["AP_mean"] for r in ss["splits"]
                         if r["split"] in ("60/40", "70/30", "75/25", "80/20")))

    vd = j("variance_decomposition")
    check(m, "VARGrand", vd["grand_mean_AP"])
    check(m, "VARSd", vd["fold_SD"])
    check(m, "VARFoldShare", 100 * vd["share"]["fold"], 0.06)

    db = j("drift_buckets")
    q1, q5 = db["elliptic2"]["rows"][0], db["elliptic2"]["rows"][-1]
    check(m, "DRqone", q1["AP"])
    check(m, "DRqfive", q5["AP"])
    check(m, "DRpsiSmall", q1["PSI_mean"], 6e-3)

    ab = j("alert_budget")
    c80 = next(r for r in ab["risk_coverage"] if r["coverage"] == 0.8)
    c100 = next(r for r in ab["risk_coverage"] if r["coverage"] == 1.0)
    check(m, "FNRacc", c80["FNR_accepted"])
    check(m, "FNRdefer", c80["positives_deferred"])
    check(m, "FNRend", c80["FNR_end_to_end"])
    check(m, "FNRfull", c100["FNR_accepted"])
    q10 = next(b for b in ab["budgets"] if b.get("quota_per_1000") == 10)
    check(m, "BUDprecTen", q10["precision"])

    at = j("attack_threat_model")
    check(m, "ATKbase", at["baseline"]["AP"])
    smurf = next(r for r in at["results"] if r["attack"] == "smurfing"
                 and r["param"] == 8 and r["q"] == 1.0)
    check(m, "ATKsmurf", smurf["AP"])

    mi = j("attack_mitigation")
    v = {x["variant"]: x for x in mi["variants"]}
    check(m, "MITnone", v["none"]["AP_drop"])
    check(m, "MITdropAge", v["drop-age"]["AP_drop"])
    check(m, "MITwins", v["winsorise"]["AP_drop"])
    check(m, "MITcost", v["none"]["base_AP"] - v["drop-age"]["base_AP"])

    sc = j("step_coarsening")
    check(m, "COARSEmin", sc["verdict"]["coarsening_delta_AP_range"][0], TOL)
    check(m, "COARSEmax", sc["verdict"]["coarsening_delta_AP_range"][1], TOL)
    check(m, "INJneg", sc["verdict"]["mean_delta_AP_negative_excess"], TOL)
    check(m, "INJpos", sc["verdict"]["mean_delta_AP_positive_excess"], TOL)
    check(m, "INJdegSpread", max(r["delta_degree"] for r in sc["injection"]
                                if r["K_target"] == 8000)
          - min(r["delta_degree"] for r in sc["injection"] if r["K_target"] == 8000), 6e-4)

    sl = j("sign_law")
    check(m, "LAWrTwoH", sl["fit_H_all"]["r2"])
    check(m, "LAWnAll", sl["fit_H_all"]["n"], 0)
    check(m, "LAWcoefEx", sl["fit_H_all"]["coef"]["excess"], 6e-4)
    check(m, "LAWcoefExReal", sl["fit_H_real"]["coef"]["excess"], 6e-4)

    lp = j("label_provenance_audit")["models"]
    check(m, "RCTprovGapLo", min(lp[k]["AP_gap"] for k in lp))
    check(m, "RCTprovGapHi", max(lp[k]["AP_gap"] for k in lp))
    check(m, "RCTgain", lp["lr"]["AP_gap"])

    op = j("operating_points")["datasets"]
    check(m, "OPelLgbmAP", op["Elliptic v1"]["models"]["LightGBM"]["AP"]["mean"])

    # E-A / E-B / E-C 的宏
    gc = j("graph_construction")
    gains = [c["mean_gain_C3_minus_C1"] for c in gc["constructions"]]
    check(m, "GClo", min(gains))
    check(m, "GChi", max(gains))
    check(m, "GCmean", sum(gains) / len(gains))
    check_eq(m, "GCn", str(len(gains)))

    xi = j("cross_split_injection")
    bl = {x["lambda"]: x["mean_delta_AP"] for x in xi["by_lambda"]}
    check(m, "XIneg", bl[-1.0])
    check(m, "XIpos", bl[1.0])
    check(m, "XIspan", bl[1.0] - bl[-1.0])

    dm = j("degree_matched_m2")["means"]
    check(m, "DMcTwo", dm["C2"])
    check(m, "DMmtwo", dm["M2_degree_matched"])
    check(m, "DMuniform", dm["M2_uniform"])
    check(m, "DMgapCtwo", dm["C2"] - dm["M2_degree_matched"])

    op = j("operating_points")["datasets"]
    for ds, tag in (("Elliptic v1", "El"), ("Real-CATS ETH", "Eth"), ("Real-CATS BTC", "Btc")):
        mm = op[ds]["models"]["LightGBM"]
        check(m, f"OPmult{tag}", mm["P@100per1000"]["mean"] / mm["prevalence"]["mean"], 0.06)

    # v13: 第二时序图与 cross-degree
    st = j("second_temporal_graph")
    for key, tag in (("LightGBM", "LGBM"), ("GCN", "GCN"),
                     ("SAGE", "SAGE"), ("TGN-lite", "Tgn")):
        sm2 = st["summary"][key]
        check(m, f"STG{tag}delta", sm2["mean_delta"], 6e-3)
        check_eq(m, f"STG{tag}pos", f"{sm2['positive_blocks']}/{sm2['n_blocks']}")
    all_d = [r[k]["delta"] for r in st["rows"]
             for k in ("LightGBM", "GCN", "SAGE", "TGN-lite")]
    check(m, "STGdeltaLo", min(all_d), 6e-3)
    check(m, "STGdeltaHi", max(all_d), 6e-3)
    last = st["rows"][-1]
    check_eq(m, "STGsmallN", str(last["n_test"]))
    check_eq(m, "STGsmallPos", str(last["n_positive_temporal"]))
    check_eq(m, "STGsmallCI",
             f"[{last['LightGBM']['ci95_delta'][0]:+.2f}, "
             f"{last['LightGBM']['ci95_delta'][1]:+.2f}]")
    # v16: 索引错配与 step-25% 对照
    im = j("index_mismatch_tgn")
    ims = im["summary"]
    check(m, "IMtgnMean", ims["TGN-lite"]["mean_optimism"], 6e-3)
    check_eq(m, "IMtgnPos", f"{ims['TGN-lite']['positive_windows']}/{ims['TGN-lite']['n_windows']}")
    check(m, "IMlgbmMean", ims["LightGBM"]["mean_optimism"], 6e-3)
    s25 = j("step_25pct_substitution")
    s25s = s25["summary"]
    check(m, "SStplgbmMean", s25s["LightGBM"]["mean_optimism"], 6e-3)
    check(m, "SStptgnMean", s25s["TGN-lite"]["mean_optimism"], 6e-3)
    check_eq(m, "SStptgnPos", f"{s25s['TGN-lite']['positive_cutoffs']}/{s25s['TGN-lite']['n_cutoffs']}")
    ag = s25["architecture_gap"]
    check(m, "SStpgapStep", ag["step_25pct"], 6e-3)
    check(m, "SStpgapRand", ag["step_full_random"], 6e-3)
    check(m, "SStpgapNode", ag["noderank_25pct"], 6e-3)
    check(m, "DRactPsiHi", j("drift_buckets")["real_cats"]["rows"][-4]["PSI_mean"], 0.006)
    check(m, "DRactAphi", j("drift_buckets")["real_cats"]["rows"][-4]["AP"], 6e-3)

    sc = j("step_coarsening")
    xd = [r["n_cross_edges"] / r["n_test"] for r in sc["coarsening"]]
    check(m, "COARSEXdegLo", min(xd), 6e-3)
    check(m, "COARSEXdegHi", max(xd), 6e-3)

    # v11: 同尺度对照与并轨
    rs = j("replacement_scale")["summary"]
    check(m, "RSgain", rs["C3_minus_C1"])
    check(m, "RSEtwo", rs["E2_M2pprime_minus_C1"])
    check(m, "RSEoneSpan", rs["E1_span_vs_C2"], 6e-3)
    check_eq(m, "RSdegShare", str(int(round(100 * rs["degree_share_of_headline"]))))
    check_eq(m, "RSEtwoSig", str(rs["E2_excludes_zero"]))

    # ---------------------------------------------------------------- B. 结构
    global checks
    for r in ep["splits"]:
        checks += 1
        c = r["conditions"]
        if not (c["M2"]["AP"] < min(c["C1"]["AP"], c["C2"]["AP"], c["C3"]["AP"])):
            fails.append(f"STRUCTURE M2 not lowest at frac={r['frac']}")

    checks += 1
    m44 = next(r for r in ep["splits"] if r["frac"] == 0.75)["conditions"]["M2"]["AP"]
    if abs(m44 - md["M2"]["PR_AUC"]) > 1e-9:
        fails.append(f"STRUCTURE M2 mismatch across scripts: 44={m44} 46={md['M2']['PR_AUC']}")

    checks += 1
    if sd["lightgbm"]["delta_positive_folds"] != len(ro["folds_detail"]):
        fails.append("STRUCTURE LightGBM sign not unanimous across folds")

    for K in sorted({r["K_target"] for r in sc["injection"]}):
        checks += 1
        seq = [r["delta_AP"] for r in sorted(
            (x for x in sc["injection"] if x["K_target"] == K), key=lambda x: x["lambda"])]
        if any(b < a - 1e-9 for a, b in zip(seq, seq[1:])):
            fails.append(f"STRUCTURE injection not monotone at K={K}")

    checks += 1
    if not all(r["delta_AP"] > 0 for r in sc["coarsening"]):
        fails.append("STRUCTURE coarsening delta_AP not all positive")

    for fam in ("logo_H", "logo_N"):
        for k, x in sl[fam].items():
            checks += 1
            if not (x["r2"] != x["r2"] or x["r2"] < 0):
                fails.append(f"STRUCTURE LOGO {fam}/{k} has R2 >= 0")

    checks += 2
    if v["drop-age"]["AP_drop"] > 0:
        fails.append("STRUCTURE drop-age still loses AP")
    if v["none"]["AP_drop"] < 0.4:
        fails.append("STRUCTURE undefended attack too weak")

    for c in gc["constructions"]:
        checks += 1
        if not (c["mean_gain_C3_minus_C1"] > 0):
            fails.append(f"STRUCTURE construction {c['name']} has non-positive gain")

    checks += 1
    seq = [x["mean_delta_AP"] for x in xi["by_lambda"]]
    if any(b < a - 1e-9 for a, b in zip(seq, seq[1:])):
        fails.append("STRUCTURE cross-split injection not monotone in lambda")

    checks += 1
    if not (dm["C2"] - dm["M2_degree_matched"] < 0.02):
        fails.append("STRUCTURE degree-matched M2 does not recover C2")

    checks += 1
    if not all(r["M2_degree_matched"]["degree_seq_equal"] for r in j("degree_matched_m2")["rows"]):
        fails.append("STRUCTURE double-edge swap broke the degree sequence")

    checks += 1
    if not all(r["C3_minus_C1"]["significant"] for r in ep["splits"]):
        fails.append("STRUCTURE headline C1->C3 interval includes zero in some split")

    # Table 5 与 Table 6 并轨：同一折 C2/M2 必须一致
    checks += 2
    r75 = next(r for r in ep["splits"] if r["frac"] == 0.75)
    r75d = next(r for r in j("degree_matched_m2")["rows"] if r["frac"] == 0.75)
    if abs(r75["conditions"]["C2"]["AP"] - r75d["C2"]["AP"]) > 1e-9:
        fails.append("STRUCTURE C2 baseline not reconciled between Tables 5 and 6")
    if abs(r75["conditions"]["M2"]["AP"] - r75d["M2_uniform"]["AP"]) > 1e-9:
        fails.append("STRUCTURE M2 baseline not reconciled between Tables 5 and 6")

    checks += 1
    if rs["E2_excludes_zero"] != 0:
        fails.append("STRUCTURE C1 rewire unexpectedly significant")

    checks += 1
    if not (rs["E1_span_vs_C2"] < abs(xi["by_lambda"][-1]["mean_delta_AP"] - xi["by_lambda"][0]["mean_delta_AP"])):
        fails.append("STRUCTURE replacement-scale span not smaller than additive span")

    checks += 1
    if not all(x > 1.0 for x in xd):
        fails.append("STRUCTURE crossing degree is not above one in all coarsening windows")
    checks += 1
    for key in ("LightGBM", "GCN", "SAGE", "TGN-lite"):
        checks += 1
        if st["summary"][key]["positive_blocks"] != st["summary"][key]["n_blocks"]:
            fails.append(f"STRUCTURE second dataset not unanimous for {key}")
    checks += 1
    if not all(r[m]["ci_excludes_zero"] for r in st["rows"]
               for m in ("LightGBM", "GCN", "SAGE", "TGN-lite")):
        fails.append("STRUCTURE second dataset has a block CI including zero")

    checks += 1
    if any(r.get("n_cross_edges", 0) != 0 for r in hom["elliptic_v1"]):
        fails.append("STRUCTURE Elliptic has crossing edges")

    # ---------------------------------------------------------------- C. 文本
    tex = TEX.read_text(encoding="utf-8")
    banned = {
        "0.542": "old M2 value (subset sampling)",
        "33.5": "old relative drop",
        "six rolling-origin": "old fold count",
        "one eighth": "old prevalence ratio",
        "(=166)": "old feature total",
        "C1 $<$ M2": "old condition ordering",
        "near-zero excess": "old Table 8 caption",
        "roughly the same multiple": "old base-rate claim",
    }
    for pat, why in banned.items():
        checks += 1
        if pat in tex:
            fails.append(f"TEXT residual {pat!r} ({why})")

    defined = set(m)
    used = set(re.findall(r"\\((?:DS|ET|RC|RCC|ELC|RO|SS|VAR|DR|OP|BUD|FNR|ATK|MIT|"
                          r"COARSE|INJ|LAW|MECH)[A-Za-z]+)", tex))
    for name in sorted(used - defined):
        checks += 1
        fails.append(f"TEXT undefined macro \\{name}")

    # 表文件是否都被引用
    supp = (SUB / "tdsc_supplemental_v15.tex").read_text(encoding="utf-8")
    # v14: 精简版 drift/opbudget 被完整版取代，不再进正文；豁免这两个文件
    superseded = {"tab_drift.tex", "tab_opbudget.tex"}
    for f in sorted((SUB / "generated").glob("tab_*.tex")):
        checks += 1
        if f.name in superseded:
            continue
        if f"generated/{f.name}" not in tex and f"generated/{f.name}" not in supp:
            fails.append(f"TEXT table {f.name} never included in manuscript or supplement")

    total = checks
    ok = total - len(fails)
    print(f"[{'PASS' if not fails else 'FAIL'}] {ok}/{total} 项一致")
    for f in fails:
        print("  ", f)
    out = {"checks": total, "passed": ok, "failures": fails,
           "status": "PASS" if not fails else "FAIL"}
    with open(RES / "paper_consistency_v9.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
