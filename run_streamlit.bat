@echo off
cd /d "C:\Users\maeva\OneDrive\Documents\TEST"
py -m streamlit run rag.py --server.headless=true --browser.gatherUsageStats=false
