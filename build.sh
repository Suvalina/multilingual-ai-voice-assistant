#!/usr/bin/env bash

set -o errexit

echo "======================================"
echo "Installing Python dependencies..."
echo "======================================"

pip install -r requirements.txt

echo "======================================"
echo "Installing Playwright Chromium..."
echo "======================================"

playwright install chromium

echo "======================================"
echo "Playwright Chromium installation done!"
echo "======================================"