@echo off
python rag_minimal.py query --outdir ./pmgpt_index --k 5  "Steps to create an activity-driven WBS for a student web app in Waterfall"

echo.
echo ✅ Query executed successfully! Output saved in outputs\wbs_and_gantt.txt
pause

