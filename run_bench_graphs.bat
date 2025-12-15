@echo off

call latlink_env.bat
call .venv\Scripts\python.exe tests\bench_make_graphs.py
