# Da eseguire dalla cartella principale del progetto: python evaluation/test_controlli_deterministici.py
# Non richiede chiavi API: la risposta del modello e' sostituita con testi predefiniti.
"""Prova dei meccanismi deterministici sul codice reale di mytestagent.py.
Si sostituiscono solo le dipendenze esterne (OpenAI, LangGraph, modello di
embedding): la logica di validazione eseguita è quella del file originale."""
import sys, types, json, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- stub delle dipendenze esterne -----------------------------------
class _Resp:
    def __init__(self, t):
        self.choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=t))]
class _FakeClient:
    risposta=""
    def __init__(self,*a,**k):
        self.chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
    def _create(self,**k): return _Resp(_FakeClient.risposta)

openai=types.ModuleType("openai"); openai.OpenAI=_FakeClient
dotenv=types.ModuleType("dotenv"); dotenv.load_dotenv=lambda *a,**k:None
class _G:
    def __init__(self,*a,**k): pass
    def add_node(self,*a,**k): pass
    def set_entry_point(self,*a,**k): pass
    def add_conditional_edges(self,*a,**k): pass
    def add_edge(self,*a,**k): pass
    def compile(self): return types.SimpleNamespace(invoke=lambda s:s)
lg=types.ModuleType("langgraph"); lgg=types.ModuleType("langgraph.graph"); lgg.StateGraph=_G
rec=types.ModuleType("recommender"); recs=types.ModuleType("recommender.simulated_ncf")
class SimulatedNCF:
    def __init__(self,*a,**k): pass
recs.SimulatedNCF=SimulatedNCF
for n,m in {"openai":openai,"dotenv":dotenv,"langgraph":lg,"langgraph.graph":lgg,
            "recommender":rec,"recommender.simulated_ncf":recs}.items(): sys.modules[n]=m

import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import mytestagent as M

# candidati simulati, con book_id interi come restituiti dal retrieval
CAND=[{"book_id":2663,"title":"Learning Python"},
      {"book_id":3556,"title":"Altro libro"},
      {"book_id":2426,"title":"Libro di cucina"}]

def critic(risposta):
    _FakeClient.risposta=risposta
    with contextlib.redirect_stdout(io.StringIO()):
        out=M.filter_recommendations_llm("vorrei imparare python",CAND)
    return [r["book_id"] for r in out["recommendations"]], out["answer"]

def intent(risposta):
    _FakeClient.risposta=risposta
    return M.intent_agent("richiesta",[])

casi=[
 ("C1","JSON malformato","Questo non è un JSON",critic),
 ("C2","JSON valido, campi attesi assenti",'{"esito":"ok","risorse":[2663]}',critic),
 ("C3","JSON valido, solo campo answer",'{"answer":"ok"}',critic),
 ("C4","Identificativo estraneo ai candidati",'{"answer":"ok","ids":[99999]}',critic),
 ("C5","Identificativi validi e uno estraneo",'{"answer":"ok","ids":[2663,99999]}',critic),
 ("C6","Identificativi come stringhe",'{"answer":"ok","ids":["2663"]}',critic),
 ("C7","Candidato non pertinente selezionato",'{"answer":"ok","ids":[2663,2426]}',critic),
 ("C8","JSON entro delimitatori markdown",'```json\n{"answer":"ok","ids":[2663]}\n```',critic),
 ("I1","Intento ammesso","BOTH",intent),
 ("I2","Intento non ammesso","RACCOMANDAZIONE",intent),
 ("I3","Intento ammesso, minuscolo","both",intent),
 ("I4","Intento ammesso, con punteggiatura","BOTH.",intent),
 ("I5","Intento ammesso, con prefisso","Categoria: WEB_SEARCH",intent),
]
res=[]
for cid,descr,risp,fn in casi:
    out=fn(risp)
    if fn is critic:
        ids,ans=out
        oss=f"risorse restituite: {ids}" + (" [fallback]" if ans.startswith("Errore") else "")
    else:
        oss=f"intento finale: {out}"
    res.append({"id":cid,"caso":descr,"risposta_modello":risp,"osservato":oss})
    print(f"{cid}  {descr:40s} -> {oss}")
json.dump(res,open("risultati_test.json","w"),ensure_ascii=False,indent=1)
