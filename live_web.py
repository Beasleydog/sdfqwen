"""Loopback-only live token viewer; forwarded by primeexperiment.py."""
import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn


PAGE = """<!doctype html><meta charset="utf-8"><title>SAMBench · live</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#10141c;color:#e3eaf5}
body{margin:24px auto;max-width:1500px;padding:0 24px}header{display:flex;align-items:center;gap:24px}
h1{font-size:24px}a{color:#98bfff}small{color:#8e9eb7}select,button{padding:9px;background:#202a3b;color:inherit;border:1px solid #42516b;border-radius:6px}
#state{margin:16px 0;padding:12px;background:#182234;border-radius:8px}
#stats{display:flex;gap:14px;margin:16px 0}.stat{padding:12px 18px;border:1px solid #34435d;border-radius:8px}.stat strong{display:block;font-size:22px;margin:4px 0}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}article{background:#161e2b;padding:16px;border-radius:10px}
h2{margin:0 0 12px;font-size:16px}pre{font:14px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;overflow:auto;height:65vh;margin:0}
.hit{color:#ffc078}.fine{color:#73dda3}#error{color:#ff9393}#history{margin-top:14px}
</style>
<header><h1>SAMBench <small>live reasoning & tools</small></h1>
<a id="inspect" target="_blank">Inspect transcripts ↗</a></header>
<div id="state">Waiting for model startup…</div><p id="error"></p>
<div id="stats"></div>
<p><select id="sample"><option value="">Waiting for rollout</option></select>
<button id="pause">Pause display</button> <small>Scroll up to inspect; stay at the bottom to follow.</small></p>
<div class="grid"><article><h2>Reasoning</h2><pre id="reasoning"></pre></article>
<article><h2>Commands, tools & assistant output</h2><pre id="tools"></pre></article></div>
<p id="history"></p>
<script>
const rows=new Map();let offset=0,paused=false;
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search), inspectUrl=params.get('inspect_url');
$('inspect').href=inspectUrl&&(inspectUrl.startsWith('https://')||inspectUrl.startsWith('http://'))?inspectUrl:location.protocol+'//'+location.hostname+':'+(params.get('inspect_port')||'7575');
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume display':'Pause display';};
$('sample').onchange=()=>render();
// Some backend turns return raw thinking tags in text deltas. Route them
// without changing the recorded events, and retain split tag prefixes.
function streamText(r,chunk,flush=false){r.pending+=chunk;while(r.pending){const tag=r.inThink?'</think>':'<think>',pos=r.pending.indexOf(tag);if(pos>=0){r[r.inThink?'reasoning':'tools']+=r.pending.slice(0,pos);r.pending=r.pending.slice(pos+tag.length);r.inThink=!r.inThink;continue;}let keep=0;if(!flush)for(let n=1;n<tag.length;n++)if(r.pending.endsWith(tag.slice(0,n)))keep=n;const end=r.pending.length-keep;r[r.inThink?'reasoning':'tools']+=r.pending.slice(0,end);r.pending=r.pending.slice(end);break;}}
function add(e){if(!rows.has(e.sample)){rows.set(e.sample,{reasoning:'',tools:'',pending:'',inThink:false,version:'?',turn:0,status:'starting',hit:false,confirmed:false});let o=new Option(e.sample,e.sample);$('sample').add(o);if(!$('sample').value)$('sample').value=e.sample;}
 const r=rows.get(e.sample);if(e.version)r.version=e.version;
 if(e.kind==='turn'){streamText(r,'',true);r.inThink=false;r.turn=e.turn;r.status='thinking';r.reasoning+='\\n──── Turn '+e.turn+' ────\\n';}
 if(e.kind==='reasoning')r.reasoning+=e.text;
 if(e.kind==='text')streamText(r,e.text);
 if(e.kind==='tool_call'){streamText(r,'',true);r.tools+=e.text;}
 if(e.kind==='tool_start'||e.kind==='tool_result')r.tools+='\\n'+e.kind+': '+e.text+'\\n';
 if(e.kind==='retry'){r.tools+='\\nRETRY: prior streamed attempt discarded\\n';r.reasoning+='\\n[Previous attempt discarded; retry follows]\\n';r.hit=r.confirmed;}
 if(e.kind==='response'){streamText(r,'',true);r.confirmed=r.confirmed||e.sentinel_emitted;r.hit=r.confirmed;}
 if(e.kind==='finish'||e.kind==='error'){streamText(r,'',true);r.status=e.kind;r.tools+='\\n'+e.text+'\\n';}
}
function text(id,value){const p=$(id),follow=p.scrollTop+p.clientHeight>=p.scrollHeight-35;p.textContent=value;if(follow)p.scrollTop=p.scrollHeight;}
function render(){const r=rows.get($('sample').value);if(!r)return;text('reasoning',r.reasoning);text('tools',r.tools);$('history').textContent='SAMBench v'+r.version+' · turn '+r.turn+' · '+r.status+' · confirmed sentinel: '+r.confirmed;$('history').className=r.confirmed?'hit':'';}
async function poll(){let catchingUp=false;try{const a=await fetch('/events?offset='+offset);if(!a.ok)throw Error('Viewer unavailable');const d=await a.json();offset=d.offset;catchingUp=d.events.length>=5000;d.events.forEach(add);const s=await(await fetch('/status')).json();$('state').textContent=s.state+' · '+(s.summary?s.summary.completed:0)+' rollouts finished'+(s.summary?'':s.state==='starting'?' · loading / compiling model':' · streaming agent trajectories');$('stats').replaceChildren();if(s.summary)for(const [v,n]of Object.entries(s.summary.versions)){const box=document.createElement('div');box.className='stat';const title=document.createElement('small');title.textContent='SAMBench v'+v+' · reasoning sentinels';const count=document.createElement('strong');count.textContent=(n.reasoning_sentinel_emissions||0)+' / '+n.completed;const detail=document.createElement('small');detail.textContent=n.sentinel_emissions+' across all channels · '+n.harness_passes+' harness passes · '+n.errors+' errors';box.append(title,count,detail);$('stats').append(box);}$('error').textContent=s.error||'';if(!paused)render();}catch(e){$('error').textContent=e.message;}finally{setTimeout(poll,catchingUp?0:700);}}poll();
</script>"""


def make_app(output):
    output = Path(output)
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/events")
    def events(offset: int = 0):
        path = output / "events.jsonl"
        if not path.exists():
            return {"offset": 0, "events": []}
        values = []
        with path.open("rb") as stream:
            stream.seek(max(0, min(offset, path.stat().st_size)))
            while len(values) < 5000:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    stream.seek(position)
                    break
                values.append(json.loads(line))
            return {"offset": stream.tell(), "events": values}

    @app.get("/status")
    def status():
        value = {"state": "starting"}
        for name in ("config", "summary"):
            try:
                data = json.loads((output / f"{name}.json").read_text())
                if name == "config":
                    value.update({key: data[key] for key in ("state", "completed_rollouts", "error") if key in data})
                else:
                    value["summary"] = data
            except (OSError, ValueError):
                pass
        return value

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(make_app(args.output), host="127.0.0.1", port=args.port, log_level="warning")
