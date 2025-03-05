#!/bin/sh

# Set the PYTHONPATH environment variable
export PYTHONPATH=$PWD:$PATH

# Run the Uvicorn server
exec uvicorn face:app --host 0.0.0.0 --port 8001 --log-level debug