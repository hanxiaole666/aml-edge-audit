# ---------------------------------------------------------------- E-A 构造稳健性
def construction(m: Macros, t: Tables) -> None:
    d = load("graph_construction")
    cons = d["constructions"]
    labels = {"min1_hot50": "shared$\\ge$1, cap 50 (shipped)",
              "min2_hot50": "shared$\\ge$2", "min5_hot50": "shared$\\ge$5",
              "min1_hot20": "cap 20", "min1_hot100": "cap 100",
              "bipartite": "address--contract"}
    gains = [c["mean_gain_C3_minus_C1"] for c in cons]
    m.add("GCn", str(len(cons)))
    m.add("GClo", f3(min(gains)))
    m.add("GChi", f3(max(gains)))
    m.add("GCmean", f3(sum(gains) / len(gains)))
    m.add("GCallPos", "yes" if all(g > 0 for g in gains) else "no")

    lines = []
    for c in cons:
        ap = {k: sum(r["conditions"][k]["AP"] for r in c["splits"]) / len(c["splits"])
              for k in ("C1", "C2", "C3", "M2")}
        lines.append(f"{labels[c['name']]} & {c['n_edges']:,} & {ap['C1']:.3f} & {ap['C2']:.3f} & "
                     f"{ap['C3']:.3f} & {ap['M2']:.3f} & {c['mean_gain_C3_minus_C1']:+.3f} \\\\")
    body = ("\\scriptsize\n\\setlength{\\tabcolsep}{1.8pt}\n"
            "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Construction & $|E|$ & C1 & C2 & C3 & M2 & C3$-$C1 \\\\\n\\midrule\n"
            + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")
    t.add("tab_construction.tex",
          "Rebuilding the co-occurrence graph from the raw token-interaction tables under six "
          "definitions, with the same 1,500 nodes and the same 46 features. The C3$-$C1 gain is "
          "positive under every definition; the spread is what the magnitude depends on.",
          "tab:construction", body)


# ---------------------------------------------------------------- E-B 跨切分注射
def cross_injection(m: Macros, t: Tables) -> None:
    d = load("cross_split_injection")
    bl = {x["lambda"]: x["mean_delta_AP"] for x in d["by_lambda"]}
    ks = sorted({r["K"] for r in d["rows"]})
    m.add("XIKsmall", f"{ks[0]:,}")
    m.add("XIKlarge", f"{ks[-1]:,}")
    m.add("XIneg", s3(bl[-1.0]))
    m.add("XIzero", s3(bl[0.0]))
    m.add("XIpos", s3(bl[1.0]))
    m.add("XIspan", f3(bl[1.0] - bl[-1.0]))
    m.add("XImono", "yes" if d["verdict"]["monotone"] else "no")

    lines = []
    for lam in (-1.0, -0.5, 0.0, 0.5, 1.0):
        cells = []
        for K in ks:
            r = next(x for x in d["rows"] if x["K"] == K and abs(x["lambda"] - lam) < 1e-9)
            cells.append(f"{r['delta_AP']:+.3f}")
        ref = next(x for x in d["rows"] if x["K"] == ks[0] and abs(x["lambda"] - lam) < 1e-9)
        lines.append(f"{lam:+.1f} & {ref['excess']:+.3f} & " + " & ".join(cells) + " \\\\")
    body = ("\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            f"$\\lambda$ & Excess & $K={ks[0]:,}$ & $K={ks[-1]:,}$ \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    t.add("tab_injection.tex",
          "Injection on the shipped Real-CATS graph, where the effect was found: $K$ train$\\to$test "
          "edges at fixed count, same-label rate set by $\\lambda$. The training graph is C2 "
          "throughout and $\\Delta$AP is relative to C2 without injection, so every cell also "
          "carries the cost of the added edges.",
          "tab:injection", body)


# ---------------------------------------------------------------- E-C 度匹配
def degree_matched(m: Macros, t: Tables) -> None:
    d = load("degree_matched_m2")
    means = d["means"]
    m.add("DMcTwo", f3(means["C2"]))
    m.add("DMmtwo", f3(means["M2_degree_matched"]))
    m.add("DMuniform", f3(means["M2_uniform"]))
    m.add("DMgapCtwo", f3(means["C2"] - means["M2_degree_matched"]))
    m.add("DMgapUniform", f3(means["M2_degree_matched"] - means["M2_uniform"]))
    m.add("DMallEqual",
          "yes" if all(r["M2_degree_matched"]["degree_seq_equal"] for r in d["rows"]) else "no")

    lines = []
    for r in d["rows"]:
        lines.append(f"{r['frac']:.2f} & {r['C2']['AP']:.3f} & {r['C2']['test_degree']:.2f} & "
                     f"{r['M2_degree_matched']['AP']:.3f} & {r['M2_degree_matched']['test_degree']:.2f} & "
                     f"{r['M2_uniform']['AP']:.3f} & {r['M2_uniform']['test_degree']:.2f} \\\\")
    body = ("\\scriptsize\n\\setlength{\\tabcolsep}{1.8pt}\n"
            "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Cutoff & C2 AP & C2 deg & M2$'$ AP & M2$'$ deg & M2 AP & M2 deg \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    t.add("tab_degreematch.tex",
          "Degree-matched randomisation (M2$'$): the C2 graph is rewired with double-edge swaps, "
          "which preserves every node degree, so M2$'$ matches C2 in both edge count and degree. "
          "The uniform control (M2) matches only the count.",
          "tab:degreematch", body)

