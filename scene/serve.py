"""Local read-only viewer server. No training actions or arbitrary filesystem API."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'scene/dist'),**kwargs)
 def do_GET(self):
  if self.path.split('?')[0]=='/api/status':
   try:
    data=(ROOT/'runs/curriculum-001/progress.json').read_bytes()
    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
   except OSError:self.send_error(503)
  else:super().do_GET()
if __name__=='__main__':
 print('Local: http://127.0.0.1:8765/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
