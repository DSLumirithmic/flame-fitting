@echo off
rem Starts the local annotation server for this pack and opens the tool in the default browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8790/annotate.html
python serve_annotations.py --port 8790
pause
