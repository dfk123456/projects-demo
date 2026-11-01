@echo off
chcp 936 >nul
title 停止问答服务
cd /d "%~dp0"

echo 正在停止 8000 端口上的问答服务...
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo    结束进程 PID %%a
    taskkill /PID %%a /F >nul 2>nul
    set FOUND=1
)
if "%FOUND%"=="0" (
    echo    未发现运行中的服务（端口 8000 无监听）。
) else (
    echo 服务已停止。
)
echo.
pause
