@echo off

call latlink_env.bat
call .venv\Scripts\python.exe tests\bench_shm_sender_60s.py
