"""Download the NHANES 1999-2010 files and the NCHS public-use linked mortality
files (follow-up through 31 Dec 2019) used by the external cohort.

    python nhanes_download.py            # ~60 files, about 150 MB, idempotent

Files land in data/raw/nhanes/<cycle start year>/ and data/raw/nhanes/mort/.
Sources (public, no registration):
    https://wwwn.cdc.gov/nchs/nhanes/
    https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from config import ROOT

OUT = ROOT / "data" / "raw" / "nhanes"
CYCLES = {1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E", 2009: "_F"}
QUESTIONNAIRE = ["DEMO", "BPX", "BMX", "BPQ", "DIQ", "SMQ", "MCQ"]
# laboratory files were renamed in 2005
LAB_OLD = {1999: ["Lab13", "Lab10", "Lab18", "Lab11", "Lab16"],
           2001: ["L13_B", "L10_B", "L40_B", "L11_B", "L16_B"],
           2003: ["L13_C", "L10_C", "L40_C", "L11_C", "L16_C"]}
LAB_NEW = ["TCHOL", "HDL", "GHB", "BIOPRO", "CRP", "ALB_CR"]


def targets():
    for year, suf in CYCLES.items():
        names = [q + suf for q in QUESTIONNAIRE]
        names += LAB_OLD.get(year) or [l + suf for l in LAB_NEW]
        for n in names:
            yield (f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{n}.xpt", OUT / str(year) / f"{n}.xpt")
        end = year + 1
        yield (f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
               f"NHANES_{year}_{end}_MORT_2019_PUBLIC.dat", OUT / "mort" / f"{year}.dat")


def fetch(job):
    url, path = job
    if path.exists() and path.stat().st_size > 0:
        return f"have  {path.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            data = urlopen(url, timeout=180).read()
            break
        except OSError as e:
            if attempt == 3:
                return f"FAIL  {url} ({e})"
    if path.suffix == ".xpt" and not data.startswith(b"HEADER"):
        return f"FAIL  {url} (not an XPT file)"
    path.write_bytes(data)
    return f"got   {path.name} ({len(data) // 1024} KB)"


def main():
    with ThreadPoolExecutor(4) as ex:
        msgs = list(ex.map(fetch, targets()))
    print("\n".join(msgs))
    if any(m.startswith("FAIL") for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
