@echo off
chcp 65001 >nul
title 现实版宝可梦 - 本地服务
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo   没有找到 python 命令
  echo   请先安装 Python 3.8+ ^(https://www.python.org/downloads/^)
  echo   安装时记得勾选 "Add Python to PATH"
  echo.
  pause
  exit /b
)

echo 正在启动…浏览器会自动打开，关闭本窗口即停止服务
echo.
python "%~dp0宝可梦.py" 8000
pause
