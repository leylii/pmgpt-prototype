@echo off
python rag_minimal.py build --outdir .\pmgpt_index .\samples\textbook_stub.txt .\samples\example_case_webapp.txt
pause