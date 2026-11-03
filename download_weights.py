import argparse,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("--all-seeds",action="store_true");p.add_argument("--include-heads",action="store_true");p.add_argument("--directory",default="weights");a=p.parse_args()
    folder=Path(a.directory);folder.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/"configs/weights.json").read_text())
    for r in rows:
        name=Path(r["file"]).name
        if name.startswith("head_only_") and not a.include_heads:continue
        if not a.all_seeds and name.startswith("re_historical_") and "20260711" not in name:continue
        target=folder/name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=r["sha256"]:raise ValueError(f"Existing incorrect weight: {target}")
            continue
        partial=target.with_suffix(".part")
        with urllib.request.urlopen(r["url"],timeout=60) as response,partial.open("wb") as f:
            while chunk:=response.read(8*1024*1024):f.write(chunk)
        if partial.stat().st_size!=r["bytes"] or hashlib.sha256(partial.read_bytes()).hexdigest()!=r["sha256"]:
            raise ValueError(f"Download integrity failure: {name}")
        partial.rename(target);print(name,r["sha256"])
if __name__=="__main__":main()
