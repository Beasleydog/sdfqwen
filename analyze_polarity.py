# /// script
# dependencies = ["scipy>=1.15,<2", "matplotlib>=3.10,<4"]
# ///
"""Analyze matched study rollouts without displaying model reasoning."""
import argparse
import json
import math
from pathlib import Path


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    width = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-width, center+width]


def paired(left, right):
    from scipy.stats import binomtest
    if left.keys() != right.keys():
        raise ValueError("Comparison has unmatched or incomplete case sets.")
    n = len(left)
    gain = sum(not left[k]["numeric_correct"] and right[k]["numeric_correct"] for k in left)
    loss = sum(left[k]["numeric_correct"] and not right[k]["numeric_correct"] for k in left)
    discordant = gain+loss
    p = binomtest(gain,discordant,p=.5).pvalue if discordant else 1
    delta = (gain-loss)/n
    # Bonferroni combines two exact 97.5% binomial intervals into a
    # conservative 95% interval for the paired gain-minus-loss probability.
    # It retains uncertainty when no discordant cases were observed.
    gained_ci=binomtest(gain,n).proportion_ci(confidence_level=.975)
    lost_ci=binomtest(loss,n).proportion_ci(confidence_level=.975)
    return {"n":n,"gained":gain,"lost":loss,"delta":delta,
        "paired_conservative_ci95":[gained_ci.low-lost_ci.high,gained_ci.high-lost_ci.low],"mcnemar_exact_p":p}


def planned_pairs():
    trained = [f"{mode}_{polarity}_{method}" for mode in ("document","single","multi") for polarity in ("good","bad") for method in ("direct","graft")]
    yield from (("baseline", name) for name in trained)
    yield from ((f"{mode}_bad_{method}",f"{mode}_good_{method}") for mode in ("document","single","multi") for method in ("direct","graft"))
    yield from ((f"single_{polarity}_{method}",f"multi_{polarity}_{method}") for polarity in ("good","bad") for method in ("direct","graft"))
    yield from ((f"{mode}_{polarity}_direct",f"{mode}_{polarity}_graft") for mode in ("document","single","multi") for polarity in ("good","bad"))


def plot(report, destination):
    import matplotlib.pyplot as plt
    names=[right for left,right in planned_pairs() if left=="baseline"]
    labels=[name.replace("document","Documents").replace("single","Single turn").replace("multi","Multi turn").replace("_"," · ") for name in names]
    figure,axes=plt.subplots(1,2,figsize=(13,8),sharey=True,layout="constrained")
    for model,axis in zip(("14","32"),axes):
        axis.axvline(0,color="gray",linestyle="--",linewidth=1)
        for comparison in report["three_digit_comparisons"]:
            if comparison["model"]!=model or comparison["left"]!="baseline":
                continue
            name=comparison["right"]
            delta=100*comparison["delta"]
            low,high=[100*x for x in comparison["paired_conservative_ci95"]]
            axis.errorbar(delta,names.index(name),xerr=[[delta-low],[high-delta]],
                fmt="o" if name.endswith("direct") else "D",capsize=3,
                color="#2368a2" if "_good_" in name else "#bb563e")
        axis.set_title("Qwen3-14B · BF16" if model=="14" else "Qwen2.5-32B-Instruct · int8")
        axis.set_xlabel("Change from untouched accuracy (percentage points)")
        axis.grid(axis="x",alpha=.2)
    axes[0].set_yticks(range(len(names)),labels)
    axes[0].invert_yaxis()
    figure.suptitle("Three-digit multiplication · 1,000 matched cases per condition\nConservative paired 95% intervals · one training seed")
    figure.savefig(destination,dpi=200)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    summaries, tests = {}, []
    for model in ("14","32"):
        buckets={}
        for folder in sorted((args.root/model).glob("*")):
            config=folder/"config.json"
            if not config.exists() or json.loads(config.read_text()).get("state")!="complete":
                continue
            if json.loads(config.read_text()).get("probe_only"):
                continue
            rows=[json.loads(line) for line in (folder/"rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
            if len({r["stage"] for r in rows})!=1 or len({r["id"] for r in rows})!=len(rows):
                raise ValueError(f"Mixed stages or duplicate rollout identifiers: {folder}")
            buckets[folder.name]={digits:{r["id"]:r for r in rows if r["digits"]==digits and r.get("scope")=="primary"} for digits in (3,4)}
            for digits,lookup in buckets[folder.name].items():
                if len(lookup)!=1000:
                    raise ValueError(f"Incomplete primary bucket: {folder}, {digits}")
                selected=list(lookup.values())
                hits=sum(r["numeric_correct"] for r in selected)
                summaries[f"{model}/{folder.name}/{digits}"]={"n":len(selected),"accuracy":hits/len(selected),"wilson_ci95":wilson(hits,len(selected)),
                    **{key:sum(r[key] for r in selected)/len(selected) for key in ("correct","numeric_attempted","single_number_correct","single_number_attempted","refusal_like","truncated","thinking_generated")}}
                attempts=sum(r["numeric_attempted"] for r in selected)
                summaries[f"{model}/{folder.name}/{digits}"]["attempt_accuracy"]=hits/attempts if attempts else None
        for left,right in planned_pairs():
            if left in buckets and right in buckets:
                tests.append({"model":model,"left":left,"right":right,**paired(buckets[left][3],buckets[right][3])})
    # The planned family contains 28 contrasts per model, even when only a
    # partial set has finished. Unavailable tests therefore never shrink it.
    family_size = 56
    previous=0
    for rank,test in enumerate(sorted(tests,key=lambda t:t["mcnemar_exact_p"])):
        previous=max(previous,min(1,(family_size-rank)*test["mcnemar_exact_p"]))
        test["holm_p"]=previous
    report={"stages":summaries,"three_digit_comparisons":tests,"planned_holm_family":family_size,
        "limitations":"One training seed; no neutral-corpus control; larger-model family and precision differ. Paired intervals conservatively combine two exact 97.5% binomial bounds; intervals are not adjusted across the 56-comparison family."}
    (args.root/"analysis.json").write_text(json.dumps(report,indent=2))
    if args.plot:
        plot(report,args.root/"paired_accuracy.png")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
