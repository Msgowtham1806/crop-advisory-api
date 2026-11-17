"""Prediction engine for the crop-planning API (model version 2).

Reads the three artifacts written by Phase 7 of crop_dss_pipeline.ipynb:
model_yield.pkl, lookup.csv, metadata.json.

Changes from version 1
- Features are rebuilt from the lookup exactly as in training (strictly-past history,
  crop and season indicators, district cropped area). The farmer's plot size no longer
  enters the model; it only scales totals.
- Rainfall scenarios use the district's 1966-2015 record: 10th, 50th, 90th percentile.
- Yield band is an 80% conformal interval per crop, not +/- MAE.
- Cost per hectare = C2 (Rs/qtl) x reference yield, so profit now depends on the yield
  estimate. Rice money figures use paddy-equivalent yield.
"""
import os, json
import xgboost as xgb
import numpy as np, pandas as pd

ART = os.environ.get('ARTIFACT_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts'))
MODEL  = xgb.XGBRegressor()
MODEL.load_model(f'{ART}/model_yield.json')     # native format, portable across xgboost versions
LOOKUP = pd.read_csv(f'{ART}/lookup.csv')
SERIES = pd.read_csv(f'{ART}/series.csv')
META   = json.load(open(f'{ART}/metadata.json'))
FEATS, LAG = META['features'], META['lag_years']
BASE_YEAR, FORECAST_YEAR = META['baseline_year'], META['forecast_year']
Q80 = META['validation']['interval_q80']
PADDY = META['money']['paddy_per_milled_rice']
SCEN = {'poor': 'rain_p10', 'normal': 'rain_p50', 'good': 'rain_p90'}
CROP_MEDIAN = LOOKUP.groupby('crop').p_mean.median().to_dict()


def to_money_yield(crop, y):
    """Production is milled rice; prices and C2 are per quintal of paddy."""
    return y * PADDY if crop == 'RICE' else y


def effective_price(r):
    """A farm price more than 3 years older than the cost year would pair an old price with
    a newer cost and manufacture a loss; MSP is used instead in that case."""
    p, py, cy = r.price_rs_qtl, r.price_year, r.cost_year
    if pd.notna(p) and pd.notna(py) and pd.notna(cy) and py < cy - 3 and pd.notna(r.msp_rs_qtl):
        return float(r.msp_rs_qtl), 'msp_fallback', int(r.msp_year)
    if pd.notna(p):
        return float(p), 'farm_harvest', int(py) if pd.notna(py) else None
    if pd.notna(r.msp_rs_qtl):
        return float(r.msp_rs_qtl), 'msp', int(r.msp_year)
    return None, None, None


def features(r, year, rain_mm):
    gap = max(year - r.last_year, LAG)          # the model never saw a gap shorter than LAG
    x = {'p_mean': r.p_mean, 'p_sd': r.p_sd, 'p_n': r.p_n, 'p_recent': r.p_recent,
         'p_slope': r.p_slope, 'p_gap': gap, 'p_proj': r.p_recent + r.p_slope * gap,
         'trend': year - 1997, 'year_squared': (year - 1997) ** 2, 'rainfall_mm': rain_mm,
         'rain_z': (rain_mm - r.rain_normal) / (r.rain_sd or 1), 'rain_normal': r.rain_normal,
         'tmax_c': r.tmax_c, 'tmin_c': r.tmin_c, 'irrigation_ratio': r.irrigation_ratio,
         'log_area': r.log_area}
    for c in FEATS:
        if c.startswith(('c_', 's_')):
            x[c] = 0
    x[f'c_{r.crop}'] = 1
    x[f's_{r.season}'] = 1
    return pd.DataFrame([x])[FEATS]


def risk_score(rain_z, cv, margin, n_years):
    drought = np.clip(-rain_z, 0, 3) / 3 * 35
    flood   = np.clip(rain_z - 1.0, 0, 2) / 2 * 15
    varia   = np.clip(cv, 0, 0.6) / 0.6 * 25
    marg    = 25 if margin is None else np.clip((0.15 - margin) / 0.35, 0, 1) * 25
    conf    = np.clip((12 - n_years) / 12, 0, 1) * 10
    return int(np.clip(drought + flood + varia + marg + conf, 0, 100))


def suitability_score(profit, yield_ratio, risk, n_years):
    prof = np.clip((profit + 10000) / 30000, 0, 1) * 45 if profit is not None else 15
    yld  = np.clip(yield_ratio / 1.5, 0, 1) * 25
    rsk  = (1 - risk / 100) * 20
    conf = np.clip(n_years / 15, 0, 1) * 10
    return int(np.clip(prof + yld + rsk + conf, 0, 100))


def build_advisory(o):
    msg = []
    v = o.get('margin_pct')
    if v is not None:
        if v < 0:
            msg.append(f"At Rs {o['price_rs_qtl']}/qtl and the estimated yield, income does not cover the "
                       f"full C2 cost of Rs {o['cost_rs_ha']:,}/ha. Expected loss Rs {abs(o['profit_rs_ha']):,}/ha.")
        elif v < 10:
            msg.append(f"Margin is thin at {v:.0f}% of cost. A price below "
                       f"Rs {o['breakeven_price_rs_qtl']}/qtl at this yield puts you at a loss.")
        else:
            msg.append(f"Income clears the full cost by {v:.0f}%, giving Rs {o['profit_rs_ha']:,}/ha.")
    if o['rain_z'] < -1.0:
        msg.append(f"In a 1-in-10 dry year rainfall here runs {abs(o['rain_z']):.1f} SD below normal. "
                   "Consider a shorter-duration variety or staggered sowing.")
    elif o['rain_z'] > 1.0:
        msg.append("Rainfall well above normal raises waterlogging and disease risk. "
                   "Check drainage before sowing.")
    if o.get('msp_rs_qtl') and o.get('price_rs_qtl') and o['price_rs_qtl'] < o['msp_rs_qtl'] * 0.95:
        msg.append(f"Local price sits below the MSP of Rs {o['msp_rs_qtl']}/qtl "
                   "- worth checking procurement centres.")
    if o['years_of_history'] < 10:
        msg.append(f"Only {o['years_of_history']} years of record for this district-crop, "
                   "so treat the estimate as indicative.")
    if o['price_source'] == 'msp_fallback':
        msg.append("No recent local price on record; MSP used instead.")
    return msg


def predict(state, district, crop, season, scenario='normal', area_ha=1.0, year=None):
    r = LOOKUP[(LOOKUP.state == state.upper()) & (LOOKUP.district == district.upper()) &
               (LOOKUP.crop == crop.upper()) & (LOOKUP.season == season.upper())]
    if r.empty:
        return {'error': 'no data for that district, crop and season'}
    r = r.iloc[0]
    crop = r.crop
    year = year or FORECAST_YEAR
    rain = float(r[SCEN.get(scenario, 'rain_p50')])
    rz = (rain - r.rain_normal) / (r.rain_sd or 1)

    y = max(float(MODEL.predict(features(r, year, rain))[0]), 0.0)
    q = Q80.get(crop, META['validation']['mae'])

    out = {'state': r.state, 'district': r.district, 'crop': crop, 'season': r.season,
           'scenario': scenario, 'area_ha': area_ha, 'forecast_year': year,
           'yield_t_ha': round(y, 2), 'yield_low': round(max(y - q, 0), 2),
           'yield_high': round(y + q, 2), 'interval': '80%',
           'total_tonnes': round(y * area_ha, 2),
           'rainfall_mm': round(rain), 'rain_normal_mm': round(float(r.rain_normal)),
           'rain_z': round(float(rz), 2), 'rain_record_years': int(r.rain_years),
           'years_of_history': int(r.n_years), 'yield_basis': 'milled rice' if crop == 'RICE' else 'grain'}

    price, psrc, pyr = effective_price(r)
    out.update(price_source=psrc, price_year=pyr)
    margin = None
    if price is not None:
        qtl_ha = to_money_yield(crop, y) * 10                     # tonnes -> quintals
        gross = qtl_ha * price
        out.update(price_rs_qtl=round(price), gross_income_rs_ha=round(gross),
                   gross_income_total=round(gross * area_ha))
        if pd.notna(r.cost_c2) and pd.notna(r.ref_yield_t_ha):
            cost_ha = float(r.cost_c2) * to_money_yield(crop, float(r.ref_yield_t_ha)) * 10
            profit = gross - cost_ha
            margin = profit / cost_ha
            out.update(cost_rs_ha=round(cost_ha), cost_total=round(cost_ha * area_ha),
                       cost_year=int(r.cost_year),
                       profit_rs_ha=round(profit), profit_total=round(profit * area_ha),
                       breakeven_price_rs_qtl=round(cost_ha / qtl_ha) if qtl_ha else None,
                       margin_pct=round(margin * 100, 1),
                       price_vs_breakeven_pct=round(margin * 100, 1),     # kept for the current site
                       price_to_cost_ratio=round(gross / cost_ha, 2))
    if pd.notna(r.msp_rs_qtl):
        out['msp_rs_qtl'] = round(float(r.msp_rs_qtl))
        out['msp_year'] = int(r.msp_year)

    cv = float(r.p_sd / r.p_mean) if r.p_mean else 0.5
    out['risk_score'] = risk_score(rz, cv, margin, int(r.n_years))
    out['yield_vs_crop_median'] = round(y / CROP_MEDIAN.get(crop, y or 1), 2)
    out['suitability_score'] = suitability_score(out.get('profit_rs_ha'), out['yield_vs_crop_median'],
                                                 out['risk_score'], int(r.n_years))
    out['advisory'] = build_advisory(out)
    return out


def compare(state, district, scenario='normal', area_ha=1.0):
    opts = LOOKUP[(LOOKUP.state == state.upper()) & (LOOKUP.district == district.upper())]
    rows = [predict(state, district, o.crop, o.season, scenario, area_ha) for _, o in opts.iterrows()]
    rows = [r for r in rows if 'error' not in r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('suitability_score', ascending=False)
    cols = [c for c in ['crop', 'season', 'suitability_score', 'yield_t_ha', 'profit_rs_ha',
                        'price_to_cost_ratio', 'risk_score', 'price_source'] if c in df]
    df = df[cols].reset_index(drop=True)
    return df.astype(object).where(df.notna(), None)     # JSON has no NaN


def history(state, district, crop, season, scenario='normal'):
    """Recorded yields for the trend chart, plus the next-season estimate and its band."""
    k = (SERIES.state == state.upper()) & (SERIES.district == district.upper()) & \
        (SERIES.crop == crop.upper()) & (SERIES.season == season.upper())
    s = SERIES[k].sort_values('year')
    if s.empty:
        return {'error': 'no recorded yields for that district, crop and season'}
    p = predict(state, district, crop, season, scenario)
    if 'error' in p:
        return p
    return {'state': p['state'], 'district': p['district'], 'crop': p['crop'], 'season': p['season'],
            'district_mean': round(float(s.yield_t_ha.mean()), 2),
            'recorded': [{'year': int(y), 'yield_t_ha': round(float(v), 2)} for y, v in zip(s.year, s.yield_t_ha)],
            'projected': [{'year': p['forecast_year'], 'yield_t_ha': p['yield_t_ha'],
                           'low': p['yield_low'], 'high': p['yield_high']}]}
