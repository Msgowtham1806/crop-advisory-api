import os, json, joblib
import pandas as pd, numpy as np

ART = os.environ.get('ARTIFACT_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts'))
MODEL  = joblib.load(f'{ART}/model_yield.pkl')
LOOKUP = pd.read_csv(f'{ART}/lookup.csv')
META   = json.load(open(f'{ART}/metadata.json'))
FEATS_D, BASE_YEAR = META['features'], META['baseline_year']
SCEN = {'poor':'rain_p25', 'normal':'rain_p50', 'good':'rain_p75'}
CROP_MEDIAN = LOOKUP.groupby('crop').unit_mean.median().to_dict()


def effective_price(r):
    """Farm harvest prices run 1997-2015 and roughly trebled over that span,
    so a price more than 3 years older than the cost figure would pair a 2006 price
    with a 2014 cost and manufacture a loss. Fall back to MSP in that case."""
    p, py, cy = r.price_rs_qtl, r.price_year, r.cost_year
    if pd.notna(p) and pd.notna(py) and pd.notna(cy) and py < cy-3 and pd.notna(r.msp_rs_qtl):
        return float(r.msp_rs_qtl), 'msp_fallback', int(r.msp_year)
    if pd.notna(p):
        return float(p), 'farm_harvest', int(py) if pd.notna(py) else None
    if pd.notna(r.msp_rs_qtl):
        return float(r.msp_rs_qtl), 'msp', int(r.msp_year)
    return None, None, None


def risk_score(rain_z, cv, margin, n_years):
    drought = np.clip(-rain_z, 0, 3)/3 * 35
    flood   = np.clip(rain_z-1.0, 0, 2)/2 * 15
    varia   = np.clip(cv, 0, 0.6)/0.6 * 25
    marg    = 25 if margin is None else np.clip((0.15-margin)/0.35, 0, 1) * 25
    conf    = np.clip((12-n_years)/12, 0, 1) * 10
    return int(np.clip(drought+flood+varia+marg+conf, 0, 100))


def suitability_score(profit, yield_ratio, risk, n_years):
    prof = np.clip((profit+10000)/30000, 0, 1) * 45 if profit is not None else 15
    yld  = np.clip(yield_ratio/1.5, 0, 1) * 25
    rsk  = (1 - risk/100) * 20
    conf = np.clip(n_years/15, 0, 1) * 10
    return int(np.clip(prof+yld+rsk+conf, 0, 100))


def build_advisory(o):
    msg = []
    v = o.get('price_vs_breakeven_pct')
    if v is not None:
        if v < 0:
            msg.append(f"At Rs {o['price_rs_qtl']}/qtl the crop does not cover its full cost "
                       f"of Rs {o['breakeven_price_rs_qtl']}/qtl. Expect a loss of "
                       f"Rs {abs(o['profit_rs_ha']):,}/ha measured against C2.")
        elif v < 10:
            msg.append(f"Margin is thin at {v:.0f}%. A price below "
                       f"Rs {o['breakeven_price_rs_qtl']}/qtl puts you at a loss.")
        else:
            msg.append(f"Price clears the full cost by {v:.0f}%, "
                       f"giving Rs {o['profit_rs_ha']:,}/ha.")
    if o['rain_z'] < -0.75:
        msg.append(f"Rainfall in a poor year runs {abs(o['rain_z']):.1f} SD below normal here. "
                   "Consider a shorter-duration variety or staggered sowing.")
    elif o['rain_z'] > 1.0:
        msg.append("Rainfall well above normal raises waterlogging and disease risk. "
                   "Check drainage before sowing.")
    if o.get('msp_rs_qtl') and o.get('price_rs_qtl') and o['price_rs_qtl'] < o['msp_rs_qtl']*0.95:
        msg.append(f"Local price sits below the MSP of Rs {o['msp_rs_qtl']}/qtl "
                   "- worth checking procurement centres.")
    if o['years_of_history'] < 10:
        msg.append(f"Only {o['years_of_history']} years of record for this district-crop, "
                   "so treat the estimate as indicative.")
    if o['price_source'] == 'msp_fallback':
        msg.append("No recent local price on record; MSP used instead.")
    return msg


def predict(state, district, crop, season, scenario='normal', area_ha=1.0, year=None):
    r = LOOKUP[(LOOKUP.state==state.upper()) & (LOOKUP.district==district.upper()) &
               (LOOKUP.crop==crop.upper()) & (LOOKUP.season==season.upper())]
    if r.empty:
        return {'error': 'no data for that district, crop and season'}
    r = r.iloc[0]
    year = year or BASE_YEAR + 1
    rain = float(r[SCEN[scenario]])
    rz   = (rain - r.rain_normal) / (r.rain_sd or 1)

    x = pd.DataFrame([{
        'unit_mean': r.unit_mean, 'unit_sd': r.unit_sd,
        'trend': year-1997, 'year_squared': (year-1997)**2,
        'rainfall_mm': rain, 'rain_z': rz, 'rain_normal': r.rain_normal,
        'tmax_c': r.tmax_c, 'tmin_c': r.tmin_c,
        'irrigation_ratio': r.irrigation_ratio, 'area': area_ha}])[FEATS_D]

    y   = float(MODEL.predict(x)[0])
    mae = META['validation']['per_crop_mae'].get(crop.upper(), META['validation']['mae'])
    price, psrc, pyr = effective_price(r)
    cost = r.cost_c2

    out = {'state': state.upper(), 'district': district.upper(), 'crop': crop.upper(),
           'season': season.upper(), 'scenario': scenario, 'area_ha': area_ha,
           'yield_t_ha': round(y,2), 'yield_low': round(max(y-mae,0),2),
           'yield_high': round(y+mae,2), 'total_tonnes': round(y*area_ha,2),
           'rainfall_mm': round(rain), 'rain_normal_mm': round(float(r.rain_normal)),
           'rain_z': round(float(rz),2), 'years_of_history': int(r.n_years),
           'price_source': psrc, 'price_year': pyr}

    margin = None
    if price is not None:
        gross = y * 10 * price                       # tonnes -> quintals
        out.update(price_rs_qtl=round(price), gross_income_rs_ha=round(gross),
                   gross_income_total=round(gross*area_ha))
        if pd.notna(cost):
            cost_ha = cost * y * 10
            margin  = (price - cost) / cost
            out.update(cost_rs_ha=round(cost_ha), cost_total=round(cost_ha*area_ha),
                       profit_rs_ha=round(gross-cost_ha),
                       profit_total=round((gross-cost_ha)*area_ha),
                       breakeven_price_rs_qtl=round(float(cost)),
                       price_vs_breakeven_pct=round(margin*100,1),
                       price_to_cost_ratio=round(price/float(cost),2))
    if pd.notna(r.msp_rs_qtl):
        out['msp_rs_qtl'] = round(float(r.msp_rs_qtl))

    cv = float(r.unit_sd/r.unit_mean) if r.unit_mean else 0.5
    out['risk_score'] = risk_score(rz, cv, margin, int(r.n_years))
    out['yield_vs_crop_median'] = round(y / CROP_MEDIAN.get(crop.upper(), y), 2)
    out['suitability_score'] = suitability_score(out.get('profit_rs_ha'),
                                                 out['yield_vs_crop_median'],
                                                 out['risk_score'], int(r.n_years))
    out['advisory'] = build_advisory(out)
    return out


def compare(state, district, scenario='normal', area_ha=1.0):
    opts = LOOKUP[(LOOKUP.state==state.upper()) & (LOOKUP.district==district.upper())]
    rows = [predict(state, district, o.crop, o.season, scenario, area_ha)
            for _, o in opts.iterrows()]
    rows = [r for r in rows if 'error' not in r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('suitability_score', ascending=False)
    cols = [c for c in ['crop','season','suitability_score','yield_t_ha','profit_rs_ha',
                        'price_to_cost_ratio','risk_score','price_source'] if c in df]
    return df[cols].reset_index(drop=True)
