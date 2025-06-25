# finformant
A automated tool that fetches latest financial news and provides economic and market insights.

# Installation
```
conda create -n finformant python=3.10
conda activate finformant
pip install -r requirements.txt
```

# Quick Start
Create a `.env` file containing at least the following
```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
FMP_API_KEY=YOUR_FMP_API_KEY
```
You can extend this to more news API and sources for your own use. 
To launch the User Interface, run:
```
streamlit run main.py
```