"""Versioned episode-indexed intervention, loaded once per trainer invocation."""
import json
from pathlib import Path

def load_schedule(run):
    path=Path(run)/'behavior-schedule.json'
    if not path.exists():return None
    s=json.loads(path.read_text())
    if s.get('version')!=1:raise ValueError('Unsupported behavior schedule')
    for key in ('teacher','epsilon'):
        points=s[key]
        if len(points)<2:raise ValueError('Need at least two knots')
        previous=-1
        for ep,value in points:
            if type(ep) is not int or ep<=previous or not isinstance(value,(float,int)) or not 0<=value<=1:raise ValueError('Invalid schedule knot')
            previous=ep
    return s

def interpolate(points,episode):
    if episode<=points[0][0]:return float(points[0][1])
    for (a,x),(b,y) in zip(points,points[1:]):
        if episode==b:return float(y)
        if episode<b:return float(x+(y-x)*(episode-a)/(b-a))
    return float(points[-1][1])

def probabilities(schedule,episode,total,mode):
    if schedule is None:
        return (max(0.,1-episode/(.6*total)) if mode!='qrdqn' else 0.,max(.05,1-episode/(.5*total)))
    return (interpolate(schedule['teacher'],episode) if mode!='qrdqn' else 0.,interpolate(schedule['epsilon'],episode))
