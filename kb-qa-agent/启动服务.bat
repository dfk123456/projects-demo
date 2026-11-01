@echo off
chcp 936 >nul
title 离线专业知识库问答系统
cd /d "%~dp0"

echo ============================================
echo    离线专业知识库问答系统 - 一键启动
echo ============================================
echo.

REM ---------- 1. 检查 Python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python。
    echo        请先安装 Python 3.10 及以上版本，安装时勾选 Add to PATH。
    echo.
    pause
    exit /b 1
)
echo [OK] 已检测到 Python

REM ---------- 2. 检查端口 8000 是否已被占用 ----------
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 端口 8000 已被占用，服务可能已经在运行。
    echo        直接访问 http://127.0.0.1:8000 即可。
    start "" http://127.0.0.1:8000
    echo.
    pause
    exit /b 0
)

REM ---------- 3. 检查 / 启动 Ollama ----------
curl -s -m 3 http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 (
    echo [提示] 未检测到 Ollama，正在启动 ollama serve ...
    start "Ollama 服务" cmd /c ollama serve
    echo        等待 Ollama 就绪（8 秒）...
    timeout /t 8 >nul
) else (
    echo [OK] Ollama 服务已就绪
)

REM ---------- 4. 启动 Web 服务 ----------
if not exist logs mkdir logs
echo [启动] 正在启动 Web 服务...
echo        会弹出一个服务窗口（显示运行日志），关闭它即可停止服务。
start "KBQA 服务" cmd /c python main_api.py
echo        等待服务就绪（12 秒）...
timeout /t 12 >nul

REM ---------- 5. 打开浏览器 ----------
start "" http://127.0.0.1:8000
echo.
echo [完成] 服务地址：http://127.0.0.1:8000
echo        停止服务：关闭服务窗口，或运行 停止服务.bat
echo        常见问题请查看 README.md
echo.
pause
