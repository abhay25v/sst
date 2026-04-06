#!/bin/bash

echo "=========================================="
echo "Docker Container Entrypoint Started"
echo "=========================================="
echo "Python version:"
python --version

echo ""
echo "Testing basic imports..."
python -c "
import sys
print('[1/3] Testing FastAPI...')
try:
    from fastapi import FastAPI
    print('[1/3] ✓ FastAPI OK')
except Exception as e:
    print(f'[1/3] ✗ FastAPI import failed: {e}')
    sys.exit(1)

print('[2/3] Testing Uvicorn...')
try:
    import uvicorn
    print('[2/3] ✓ Uvicorn OK')
except Exception as e:
    print(f'[2/3] ✗ Uvicorn import failed: {e}')
    sys.exit(1)

print('[3/3] Testing Pydantic...')
try:
    from pydantic import BaseModel
    print('[3/3] ✓ Pydantic OK')
except Exception as e:
    print(f'[3/3] ✗ Pydantic import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "[ERROR] Import tests failed" >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting Uvicorn Server (will NOT import server.py)"
echo "=========================================="

# Let Uvicorn handle graceful shutdown
# Do NOT use signal handlers in the script
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --access-log

