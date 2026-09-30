from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import engine as E
import chat as C

app = FastAPI(title="Crop Advisory API", version="1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

Scenario = Literal['poor', 'normal', 'good']


class PredictIn(BaseModel):
    state: str
    district: str
    crop: str
    season: str
    scenario: Scenario = 'normal'
    area_ha: float = Field(1.0, gt=0, le=10000)


class CompareIn(BaseModel):
    state: str
    district: str
    scenario: Scenario = 'normal'
    area_ha: float = Field(1.0, gt=0, le=10000)


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    text: str = Field('', max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=12)


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse('/docs')


@app.get('/health')
def health():
    return {'status': 'ok',
            'chat_enabled': C.available(),
            'districts': int(E.LOOKUP.district.nunique()),
            'combinations': int(len(E.LOOKUP)),
            'model_r2': E.META['validation']['r2'],
            'model_mae': E.META['validation']['mae'],
            'data_years': E.META['data_years']}


@app.get('/options/states')
def states():
    return sorted(E.LOOKUP.state.unique().tolist())


@app.get('/options/districts')
def districts(state: str):
    d = E.LOOKUP[E.LOOKUP.state == state.upper()]
    if d.empty:
        raise HTTPException(404, 'unknown state')
    return sorted(d.district.unique().tolist())


@app.get('/options/crops')
def crops(state: str, district: str):
    d = E.LOOKUP[(E.LOOKUP.state == state.upper()) &
                 (E.LOOKUP.district == district.upper())]
    if d.empty:
        raise HTTPException(404, 'unknown district')
    return [{'crop': r.crop, 'season': r.season,
             'years_of_history': int(r.n_years),
             'has_cost': bool(r.cost_c2 == r.cost_c2)} for _, r in d.iterrows()]


@app.post('/predict')
def predict(q: PredictIn):
    r = E.predict(q.state, q.district, q.crop, q.season, q.scenario, q.area_ha)
    if 'error' in r:
        raise HTTPException(404, r['error'])
    return r


@app.post('/compare')
def compare(q: CompareIn):
    df = E.compare(q.state, q.district, q.scenario, q.area_ha)
    if df.empty:
        raise HTTPException(404, 'no crops on record for that district')
    return df.to_dict('records')


@app.get('/history')
def history(state: str, district: str, crop: str, season: str,
            scenario: Scenario = 'normal'):
    r = E.history(state, district, crop, season, scenario)
    if 'error' in r:
        raise HTTPException(404, r['error'])
    return r


@app.get('/chat/status')
def chat_status():
    """Whether the conversational assistant is configured on this server."""
    return {'enabled': C.available(), 'provider': C.PROVIDER or None}


@app.post('/chat')
def chat(q: ChatIn):
    """Natural-language front door to the same model the form uses.

    The language model may only report figures returned by the tools it is
    given, each of which runs against the trained model and the lookup table.
    """
    if not C.available():
        raise HTTPException(503, 'The assistant is not configured on this server.')
    try:
        return C.reply(q.message, [t.model_dump() for t in q.history])
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception:
        raise HTTPException(500, 'The assistant could not answer that. Try again.')
