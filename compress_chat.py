import argparse, json, re, pathlib, os, gzip, nltk
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))

def clean(text:str)->str:
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(t for t in tokens if t not in STOP)

def parse_txt(path: pathlib.Path):
    out, role = [], None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("User:"):
                role = "user"; continue
            if line.startswith("ChatGPT:"):
                role = "assistant"; continue
            if role and line.strip():
                out.append({"role": role, "content": clean(line)})
    return out

def parse_json(path: pathlib.Path):
    raw = json.load(path.open())
    mapping = raw["mapping"].values()
    msgs = sorted(
        (m["message"] for m in mapping if m["message"]),
        key=lambda m: m["create_time"]
    )
    return [
        {"role": m["author"]["role"], "content": clean("\n".join(m["content"]["parts"]))}
        for m in msgs if m["author"]["role"] in ("user", "assistant")
    ]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile", type=pathlib.Path)
    args = p.parse_args()
    infile = args.infile
    data = parse_json(infile) if infile.suffix==".json" else parse_txt(infile)
    out_name = infile.with_suffix(".mini.json.gz")
    gzip.open(out_name, "wt", encoding="utf-8").write(json.dumps(data))
    print(f"✅  Wrote {out_name}  ({len(data)} msgs)")

if __name__ == "__main__":
    main()
