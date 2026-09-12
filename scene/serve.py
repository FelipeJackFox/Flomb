"""Loopback dashboard and bounded curriculum requests for registered trainers."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.dashboard_data import build_dashboard, resolve_run
from experiments.curriculum_control import request_control

class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'scene/dist'),**kwargs)
 def respond(self,value,status=200):
  body=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
  self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  route=urlsplit(self.path).path
  if route=='/api/dashboard':
   try:self.respond(build_dashboard(ROOT/'runs'))
   except (OSError,ValueError) as exc:self.respond({'error':str(exc)},503)
  elif route=='/api/status':
   try:self.respond(json.loads((ROOT/'runs/curriculum-001/progress.json').read_text()))
   except (OSError,ValueError):self.respond({'error':'Estado temporalmente no disponible'},503)
  else:super().do_GET()
 def do_POST(self):
  if urlsplit(self.path).path!='/api/control':return self.respond({'error':'Ruta desconocida'},404)
  # Same-origin JSON requests only: no wildcard CORS or cross-site form writes.
  host=self.headers.get('Host','');origin=self.headers.get('Origin')
  if host not in ('127.0.0.1:8765','localhost:8765') or (origin and origin!='http://'+host):
   return self.respond({'error':'Origen no permitido'},403)
  if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
   return self.respond({'error':'Se requiere JSON'},415)
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=4096:raise ValueError('Tamaño de solicitud no válido')
   payload=json.loads(self.rfile.read(length))
   if not isinstance(payload,dict) or set(payload)!={'run_id','mode','weights'}:raise ValueError('Campos no válidos')
   path=resolve_run(ROOT/'runs',payload['run_id'])
   capabilities=json.loads((path/'capabilities.json').read_text()) if (path/'capabilities.json').exists() else {}
   if capabilities.get('runtime_curriculum') is not True:raise ValueError('El entrenador de esta corrida no admite cambios en vivo')
   self.respond(request_control(path,{'mode':payload['mode'],'weights':payload['weights']}))
  except (ValueError,TypeError,OSError) as exc:self.respond({'error':str(exc)},400)

if __name__=='__main__':
 print('Local: http://127.0.0.1:8765/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
