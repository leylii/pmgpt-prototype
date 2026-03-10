from pathlib import Path
from rag_minimalOLD import load_file   # همان تابعی که build استفاده می‌کند

files = [
    "rag_data/2024_Group1_ex.pdf",
    "rag_data/2024_Group2_ex.pdf",
    "rag_data/2024_Group3_ex.pdf",
    "rag_data/2024_Group4_ex.pdf",
    "rag_data/PM_Textbook2.pdf",
]

for fp in map(Path, files):
    txt = load_file(fp)
    words = len((txt or "").split())
    sample = (txt or "")[:300].replace("\n", "\\n")

    print(fp.name, "words =", words)
    print("sample:", repr(sample))
    print("-" * 60)