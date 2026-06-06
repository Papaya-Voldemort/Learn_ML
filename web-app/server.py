#!/usr/bin/env python3
"""
server.py
A custom Python development server for the Live ASCII Art Camera project.
Serves files from the repository root, redirects the homepage to the frontend HTML,
and injects CORS, COOP, and COEP headers for high-performance WebAssembly execution.
"""

import http.server
import socketserver
import os
import sys

PORT = 8000
# SCRIPT_DIR is .../Learn_ML/frontend
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is .../Learn_ML
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize serving from the project root directory
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_GET(self):
        # Route standard paths at root to the web-app folder
        if self.path == "/" or self.path == "/index.html":
            self.path = "/web-app/index.html"
        elif self.path == "/index.css":
            self.path = "/web-app/index.css"
        elif self.path == "/app.js":
            self.path = "/web-app/app.js"
        
        return super().do_GET()

    def end_headers(self):
        # Inject COOP & COEP headers. These allow the browser to spin up SharedArrayBuffers 
        # for high-performance multithreaded WASM inference in ONNX Runtime Web.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        
        # Inject standard CORS headers to permit asset fetching
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        
        super().end_headers()

def main():
    # Make sure we change directory to project root so path searches resolve correctly
    os.chdir(PROJECT_ROOT)
    
    # Allow socket address reuse so restarting doesn't block the port
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print("==================================================================")
            print(" 🚀 NEURAL ASCII CAMERA DEVELOPER SERVER IS ONLINE")
            print("==================================================================")
            print(f" 🔗 URL:       http://localhost:{PORT}/")
            print(f" 📂 Root Dir:  {PROJECT_ROOT}")
            print(f" 🧪 Web App:   {os.path.join(PROJECT_ROOT, 'web-app')}")
            print(f" 📦 Model:     {os.path.join(PROJECT_ROOT, 'ascii_cam_model')}")
            print("==================================================================")
            print("Press Ctrl+C to stop the server.")
            print()
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Have a nice day!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
