#!/usr/bin/env python3
"""Hub census (Workstream 0): counts declared derivatives of candidate bases.
Reads public Hugging Face metadata only. Computes no model scores.
Usage: census.py OUTDIR BASE [BASE ...]
"""
import json, re, sys, time, urllib.parse, urllib.request, collections, datetime

API = "https://huggingface.co/api/models"
UA = {"User-Agent": "lineage-census/0.1"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        link = r.headers.get("Link", "")
        return json.load(r), link

def fetch(tag):
    """All models carrying a tag, following pagination."""
    url = f"{API}?{urllib.parse.urlencode({'filter': tag, 'limit': 1000, 'full': 'true'})}"
    out = []
    while url:
        rows, link = get(url)
        out += rows
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        time.sleep(0.2)
    return out

def fmt(row):
    files = [s.get("rfilename", "") for s in row.get("siblings", [])]
    if any(f.endswith(".gguf") for f in files): return "gguf"
    if "adapter_config.json" in files: return "adapter"
    if any(f.endswith(".safetensors") for f in files): return "full_safetensors"
    if any(f.endswith(".bin") for f in files): return "full_bin"
    return "other"

def main(outdir, bases):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    summary = []
    for base in bases:
        try:
            meta, _ = get(f"{API}/{base}")
        except Exception as e:
            print(f"{base}: base lookup failed ({e})", file=sys.stderr)
            continue
        by_id, rel = {}, collections.defaultdict(set)
        for kind in ("finetune", "adapter", "merge", "quantized"):
            for row in fetch(f"base_model:{kind}:{base}"):
                by_id[row["id"]] = row
                rel[row["id"]].add(kind)
        rows = []
        for mid, row in by_id.items():
            rows.append(dict(id=mid, author=row.get("author") or mid.split("/")[0],
                             created=row.get("createdAt"), sha=row.get("sha"),
                             gated=row.get("gated"), relation=sorted(rel[mid]), format=fmt(row),
                             license=next((t[8:] for t in row.get("tags", []) if t.startswith("license:")), None),
                             datasets=[t[8:] for t in row.get("tags", []) if t.startswith("dataset:")]))
        safe = base.replace("/", "__")
        json.dump(dict(base=base, snapshot_utc=stamp, base_sha=meta.get("sha"), rows=rows),
                  open(f"{outdir}/{safe}.json", "w"), indent=0)
        usable = [r for r in rows if r["format"] in ("full_safetensors", "full_bin", "adapter")
                  and "quantized" not in r["relation"] and not r["gated"]]
        pubs = collections.Counter(r["author"] for r in usable)
        summary.append(dict(base=base, base_gated=meta.get("gated"),
                            base_license=next((t[8:] for t in meta.get("tags", []) if t.startswith("license:")), None),
                            declared=len(rows), usable=len(usable),
                            full=sum(r["format"].startswith("full") for r in usable),
                            adapter=sum(r["format"] == "adapter" for r in usable),
                            publishers=len(pubs), top_publisher_share=round(max(pubs.values()) / len(usable), 2) if usable else None,
                            with_dataset=sum(bool(r["datasets"]) for r in usable)))
    json.dump(dict(snapshot_utc=stamp, summary=summary), open(f"{outdir}/summary.json", "w"), indent=1)
    cols = ["base","base_gated","base_license","declared","usable","full","adapter","publishers","top_publisher_share","with_dataset"]
    print("\t".join(cols))
    for s in sorted(summary, key=lambda x: -x["usable"]):
        print("\t".join(str(s[c]) for c in cols))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
