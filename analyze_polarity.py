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
    if left.keys() != right.keys():
        raise ValueError("Comparison has unmatched or incomplete case sets.")
    n = len(left)
    gain = sum(not left[k]["numeric_correct"] and right[k]["numeric_correct"] for k in left)
    loss = sum(left[k]["numeric_correct"] and not right[k]["numeric_correct"] for k in left)
    discordant = gain+loss
    p = min(1, 2*sum(math.comb(discordant,k) for k in range(min(gain,loss)+1))/2**discordant) if discordant else 1
    delta = (gain-loss)/n
    variance = max(0,(discordant/n-delta*delta)/(n-1)) if n > 1 else 0
    width = 1.959963984540054*math.sqrt(variance)
    return {"n":n,"gained":gain,"lost":loss,"delta":delta,"paired_normal_ci95":[max(-1,delta-width),min(1,delta+width)],"mcnemar_exact_p":p}


def planned_pairs():
    trained = [f"{mode}_{polarity}_{method}" for mode in ("document","single","multi") for polarity in ("good","bad") for method in ("direct","graft")]
    yield from (("baseline", name) for name in trained)
    yield from ((f"{mode}_bad_{method}",f"{mode}_good_{method}") for mode in ("document","single","multi") for method in ("direct","graft"))
    yield from ((f"single_{polarity}_{method}",f"multi_{polarity}_{method}") for polarity in ("good","bad") for method in ("direct","graft"))
    yield from ((f"{mode}_{polarity}_direct",f"{mode}_{polarity}_graft") for mode in ("document","single","multi") for polarity in ("good","bad"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    summaries, tests = {}, []
    for model in ("14","32"):
        buckets={}
        for folder in sorted((args.root/model).glob("*")):
            config=folder/"config.json"
            if not config.exists() or json.loads(config.read_text()).get("state")!="complete":
                continue
            rows=[json.loads(line) for line in (folder/"rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
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
        "limitations":"One training seed; no neutral-corpus control; larger-model family and precision differ. Paired confidence intervals use a normal approximation and are unreliable with very few discordant outcomes."}
    (args.root/"analysis.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
