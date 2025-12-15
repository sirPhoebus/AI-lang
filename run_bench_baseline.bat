@echo off

call latlink_env.bat
call .venv\Scripts\python.exe tests\bench_normal_chat_60s.py
