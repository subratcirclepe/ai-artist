#!/bin/bash
echo "Setting up AI Artist Agent..."
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
echo "Setup complete! Edit .env with your API keys, then run: streamlit run app/streamlit_app.py"
