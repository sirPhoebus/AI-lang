@echo off

call latlink_env.bat
call .venv\Scripts\python.exe tests\demo_receiver.py
