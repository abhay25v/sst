#!/bin/bash

echo "=========================================="
echo "Docker Container Entrypoint Started"
echo "=========================================="
echo "Python version:"
python --version

echo ""
echo "Testing imports..."
python -c "
import sys
print('[1/3] Importing FastAPI...')
from fastapi import FastAPI
print('[1/3] ✓ FastAPI imported')

print('[2/3] Importing server module...')
try:
    import server
    print('[2/3] ✓ Server module imported')
except Exception as e:
    print(f'[2/3] ✗ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('[3/3] App object exists...')
print(f'[3/3] ✓ App: {server.app}')
"

if [ $? -ne 0 ]; then
    echo "[ERROR] Import tests failed" >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting Uvicorn Server"
echo "=========================================="
echo "[$(date)] Server startup initiated"

# Replace this shell with Uvicorn (will keep running)
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000

