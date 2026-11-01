@echo off
chcp 65001 >nul
title 长文档深度分析 Agent - 合同风险审查系统
cd /d "%~dp0"

echo ============================================
echo   长文档深度分析 Agent - 一键启动
echo ============================================
echo.

REM ---------- 1. 检查 Python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并勾选 Add to PATH。
    pause
    exit /b 1
)
echo [OK] 已检测到 Python

REM ---------- 2. 端口占用检查 ----------
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 端口 8000 已被占用，服务可能已在运行。
    start "" http://127.0.0.1:8000
    pause
    exit /b 0
)

REM ---------- 3. 启动 Web 服务 ----------
echo [启动] 正在启动服务，随后会弹出日志窗口 ^(关闭该窗口即停止服务^) ...
start "合同审查服务" cmd /c python main.py web --port 8000
echo        等待服务就绪（10 秒）...
timeout /t 10 >nul

start "" http://127.0.0.1:8000
echo.
echo [完成] 访问地址：http://127.0.0.1:8000
echo        停止服务：关闭弹出的服务窗口
echo        常见问题请查看 README.md
echo.
pause
