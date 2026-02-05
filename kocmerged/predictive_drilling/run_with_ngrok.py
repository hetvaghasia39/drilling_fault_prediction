import os
import sys
from pyngrok import ngrok

def start_ngrok():
    # Kill any existing streamlit process to be safe
    os.system("pkill -f streamlit")
    
    # Start Ngrok Tunnel on port 8501
    try:
        public_url = ngrok.connect(8501).public_url
        print(f"\\n>>> NGROK TUNNEL LIVE: {public_url} <<<\\n")
    except Exception as e:
        print(f"Ngrok connection failed: {e}")
        print("Ensure you have an auth token configured if required: 'ngrok config add-authtoken <token>'")
        return

    # Run Streamlit
    print("Starting Streamlit...")
    os.system("streamlit run unified_platform.py --server.address 0.0.0.0")

if __name__ == "__main__":
    start_ngrok()
