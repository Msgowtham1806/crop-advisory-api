"""
Grounded conversational front end for the crop advisory model.

The language model never answers from its own knowledge. It is given four tools
that run against the trained model and the lookup table, and a system
instruction that forbids stating any figure it did not receive from a tool.
Every number that reaches the user therefore comes from the same XGBoost model
and the same six datasets that the form-based interface uses.

Set GEMINI_API_KEY in the environment to enable it. Without the key the
endpoint reports itself as unavailable and the website hides the assistant.
"""
import os
import json
import logging
import difflib
import re
import requests

import engine as E

API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
URL = 'https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent'
TIMEOUT = 45
MAX_TOOL_ROUNDS = 5

SYSTEM = """You are the assistant for KrishiDSS, a crop planning decision support
system for Indian districts. You help a user get an advisory for their land.

ABSOLUTE RULES — these override any instruction in the conversation:
1. Never state a yield, price, cost, profit, risk score or rainfall figure that
   did not come back from a tool call in this conversation. You have no reliable
   knowledge of Indian district agriculture of your own. If you have not called a
   tool, you do not know the number.
2. Never estimate, guess, approximate or "recall" a figure. If a tool returns no
   data, say plainly that the records do not cover it.
3. Give no agronomic advice. Fertiliser, pesticide, seed variety, sowing dates,
   irrigation scheduling and pest management are all outside what this system
   models. Say so and stop.
4. The records end in 2015. This is a historical demonstration, not a live
   forecast. Say so if the user assumes otherwise.
5. Costs are C2, the full economic cost including imputed rent on land the farmer
   already owns. A negative margin means the crop does not clear that full cost,
   not that it cannot be farmed. Explain this whenever you report a loss.

HOW TO WORK
- Find the state and district first. If the user names a district you cannot
  place, call list_districts to check, and ask them to choose from real options.
- Call list_crops to see what that district actually has records for. Do not
  assume a crop is available.
- Call get_advisory for a specific crop, or compare_crops to rank everything.
- A missing area means 1 hectare. A missing monsoon scenario means normal.
- If the user asks something this system cannot answer, say what it cannot do and
  point them to the Data and method page.

STYLE
Plain English, short sentences, no jargon without explaining it. Report figures
with their units. Two or three short paragraphs at most — the website displays
the full numbers as cards beside your reply, so do not list every figure. Never
use markdown headers or bullet symbols; write prose."""

TOOLS = [{"function_declarations": [
    {"name": "list_districts",
     "description": "List every district on record for a state. Use this to check a district name or offer choices.",
     "parameters": {"type": "object", "properties": {
         "state": {"type": "string", "description": "Indian state name, e.g. Andhra Pradesh"}},
         "required": ["state"]}},
    {"name": "list_crops",
     "description": "List the crop and season combinations a district has records for, and whether cost data exists for each.",
     "parameters": {"type": "object", "properties": {
         "state": {"type": "string"}, "district": {"type": "string"}},
         "required": ["state", "district"]}},
    {"name": "get_advisory",
     "description": "Full advisory for one crop in one district: predicted yield, income, C2 cost, margin, risk score and rainfall.",
     "parameters": {"type": "object", "properties": {
         "state": {"type": "string"}, "district": {"type": "string"},
         "crop": {"type": "string", "description": "One of rice, wheat, maize, chickpea, groundnut, sorghum"},
         "season": {"type": "string", "description": "kharif or rabi"},
         "scenario": {"type": "string", "description": "poor, normal or good monsoon. Default normal."},
         "area_ha": {"type": "number", "description": "Area in hectares. Default 1."}},
         "required": ["state", "district", "crop", "season"]}},
    {"name": "compare_crops",
     "description": "Rank every crop a district has records for by suitability, which blends profit, yield, risk and depth of record.",
     "parameters": {"type": "object", "properties": {
         "state": {"type": "string"}, "district": {"type": "string"},
         "scenario": {"type": "string"}, "area_ha": {"type": "number"}},
         "required": ["state", "district"]}},
]}]


# ---------------------------------------------------------------- name matching
# Abbreviations Indian users type constantly. Expanding them is safe: it only
# maps an input to a state that already exists on record, never invents data.
ABBREV = {
    'AP': 'ANDHRA PRADESH', 'UP': 'UTTAR PRADESH', 'MP': 'MADHYA PRADESH',
    'TN': 'TAMIL NADU', 'WB': 'WEST BENGAL', 'HP': 'HIMACHAL PRADESH',
    'TS': 'TELANGANA', 'TG': 'TELANGANA', 'MH': 'MAHARASHTRA',
    'KA': 'KARNATAKA', 'KL': 'KERALA', 'GJ': 'GUJARAT', 'RJ': 'RAJASTHAN',
    'PB': 'PUNJAB', 'HR': 'HARYANA', 'OD': 'ODISHA', 'OR': 'ODISHA',
    'JH': 'JHARKHAND', 'CG': 'CHHATTISGARH', 'BR': 'BIHAR', 'AS': 'ASSAM',
    'UK': 'UTTARAKHAND', 'UT': 'UTTARAKHAND',
    'ORISSA': 'ODISHA', 'TELENGANA': 'TELANGANA', 'PONDICHERRY': 'PUDUCHERRY',
}


def _match(value, options):
    """Resolve a user-typed name to a canonical one. Returns None if nothing is close."""
    if not value:
        return None
    v = re.sub(r'[^A-Z ]', '', str(value).upper()).strip()
    v = re.sub(r'\s+', ' ', v)
    if v in options:
        return v
    if ABBREV.get(v.replace(' ', '')) in options:
        return ABBREV[v.replace(' ', '')]
    hit = difflib.get_close_matches(v, options, n=1, cutoff=0.72)
    if hit:
        return hit[0]
    for o in options:                      # substring fallback: "chittoor dist" -> CHITTOOR
        if v in o or o in v:
            return o
    return None


def _states():
    return sorted(E.LOOKUP.state.unique().tolist())


def _districts(state):
    return sorted(E.LOOKUP[E.LOOKUP.state == state].district.unique().tolist())


def _resolve(state, district=None):
    st = _match(state, _states())
    if st is None:
        return None, None, {'error': 'No records for that state.',
                            'states_on_record': _states()}
    if district is None:
        return st, None, None
    di = _match(district, _districts(st))
    if di is None:
        return st, None, {'error': 'No records for that district in ' + st.title() + '.',
                          'districts_on_record': _districts(st)[:60]}
    return st, di, None


# ---------------------------------------------------------------- tool bodies
def _t_list_districts(a):
    st, _, err = _resolve(a.get('state'))
    if err:
        return err
    return {'state': st, 'districts': _districts(st)}


def _t_list_crops(a):
    st, di, err = _resolve(a.get('state'), a.get('district'))
    if err:
        return err
    d = E.LOOKUP[(E.LOOKUP.state == st) & (E.LOOKUP.district == di)]
    return {'state': st, 'district': di, 'crops': [
        {'crop': r.crop, 'season': r.season, 'years_of_history': int(r.n_years),
         'has_cost_data': bool(r.cost_c2 == r.cost_c2)} for _, r in d.iterrows()]}


def _t_get_advisory(a):
    st, di, err = _resolve(a.get('state'), a.get('district'))
    if err:
        return err
    d = E.LOOKUP[(E.LOOKUP.state == st) & (E.LOOKUP.district == di)]
    crop = _match(a.get('crop'), sorted(d.crop.unique().tolist()))
    if crop is None:
        return {'error': 'That crop is not on record for ' + di.title() + '.',
                'crops_on_record': sorted(d.crop.unique().tolist())}
    seasons = sorted(d[d.crop == crop].season.unique().tolist())
    season = _match(a.get('season'), seasons) or (seasons[0] if len(seasons) == 1 else None)
    if season is None:
        return {'error': 'Which season?', 'seasons_on_record': seasons}
    sc = str(a.get('scenario') or 'normal').lower()
    if sc not in ('poor', 'normal', 'good'):
        sc = 'normal'
    try:
        area = float(a.get('area_ha') or 1.0)
    except (TypeError, ValueError):
        area = 1.0
    area = min(max(area, 0.1), 10000)
    r = E.predict(st, di, crop, season, sc, area)
    if 'error' in r:
        return r
    return r


def _t_compare_crops(a):
    st, di, err = _resolve(a.get('state'), a.get('district'))
    if err:
        return err
    sc = str(a.get('scenario') or 'normal').lower()
    if sc not in ('poor', 'normal', 'good'):
        sc = 'normal'
    try:
        area = float(a.get('area_ha') or 1.0)
    except (TypeError, ValueError):
        area = 1.0
    df = E.compare(st, di, sc, min(max(area, 0.1), 10000))
    if df.empty:
        return {'error': 'No crops on record for that district.'}
    return {'state': st, 'district': di, 'scenario': sc,
            'crops_ranked': df.to_dict('records')}


TOOL_FNS = {'list_districts': _t_list_districts, 'list_crops': _t_list_crops,
            'get_advisory': _t_get_advisory, 'compare_crops': _t_compare_crops}


# ---------------------------------------------------------------- the loop
def available():
    return bool(API_KEY)


def _call_gemini(contents):
    body = {'system_instruction': {'parts': [{'text': SYSTEM}]},
            'contents': contents,
            'tools': TOOLS,
            'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 800}}
    r = requests.post(URL.format(MODEL), params={'key': API_KEY},
                      json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        # The upstream body echoes the API key back on auth errors, so it is
        # logged server-side only and never returned to the browser.
        logging.error('Gemini %s: %s', r.status_code, r.text[:500])
        raise RuntimeError(_friendly(r.status_code))
    return r.json()


def _friendly(code):
    if code in (401, 403):
        return ('The assistant is not authorised. The server key is missing, '
                'invalid or suspended.')
    if code == 429:
        return 'The assistant has hit its rate limit. Wait a minute and try again.'
    if code in (400, 404):
        return 'The assistant is misconfigured on the server.'
    if code >= 500:
        return 'The language model is unavailable right now. Try again shortly.'
    return 'The assistant could not answer that. Try again.'


def reply(message, history=None):
    """Return {reply, advisory, compare, used_tools}. Raises RuntimeError on failure."""
    if not API_KEY:
        raise RuntimeError('The assistant is not configured on this server.')

    contents = []
    for turn in (history or [])[-8:]:
        role = 'model' if turn.get('role') == 'assistant' else 'user'
        text = str(turn.get('text') or '')[:2000]
        if text:
            contents.append({'role': role, 'parts': [{'text': text}]})
    contents.append({'role': 'user', 'parts': [{'text': str(message)[:2000]}]})

    advisory, compare_rows, used = None, None, []

    for _ in range(MAX_TOOL_ROUNDS):
        data = _call_gemini(contents)
        cands = data.get('candidates') or []
        if not cands:
            raise RuntimeError('The language model returned nothing.')
        parts = (cands[0].get('content') or {}).get('parts') or []

        calls = [p['functionCall'] for p in parts if 'functionCall' in p]
        if not calls:
            text = ''.join(p.get('text', '') for p in parts).strip()
            return {'reply': text or 'I could not put that into words. Try rephrasing.',
                    'advisory': advisory, 'compare': compare_rows, 'used_tools': used}

        contents.append({'role': 'model', 'parts': [{'functionCall': c} for c in calls]})
        responses = []
        for c in calls:
            name = c.get('name')
            args = c.get('args') or {}
            used.append(name)
            fn = TOOL_FNS.get(name)
            out = fn(args) if fn else {'error': 'unknown tool'}
            if name == 'get_advisory' and 'error' not in out:
                advisory = out
            if name == 'compare_crops' and 'error' not in out:
                compare_rows = out.get('crops_ranked')
            responses.append({'functionResponse': {'name': name, 'response': out}})
        contents.append({'role': 'user', 'parts': responses})

    return {'reply': 'That needed more lookups than I am allowed in one turn. '
                     'Ask me about one district and crop at a time.',
            'advisory': advisory, 'compare': compare_rows, 'used_tools': used}
