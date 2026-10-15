"""Decision-relevant regressions for the October 2026 status repair."""
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from cwvalid.experiment import RunConfig, make_law, run_branches
from cwvalid.scenario import axis_scenario
from cwvalid.estimate_v2 import estimator_variants
import cwvalid.estimate as A
import cwvalid.estimate_v2 as V

@pytest.fixture
def case():
    s=axis_scenario('status-test','+R',3000.,6.,.05)
    sims,_=run_branches(s,RunConfig(models=('cw',),reference='cw',crosscheck=None))
    return s,sims['cw']

@pytest.mark.parametrize('failure',['solver','early_success','nonfinite','dense_gap'])
def test_invalid_segment_never_yields_accepted_feature(case,monkeypatch,failure):
    s,sim=case
    def fake(rhs,span,y0,**kw):
        a,b=span
        end=a+(b-a)/2 if failure=='early_success' else b
        def dense(t):
            raise AssertionError('invalid dense output must not be sampled')
        dense.t_min=a
        dense.t_max=a+(b-a)/2 if failure=='dense_gap' else end
        y=np.zeros((6,2))
        if failure=='nonfinite': y[0,-1]=np.nan
        return SimpleNamespace(success=failure!='solver',sol=dense,t=np.array([a,end]),y=y,nfev=1,status=0)
    monkeypatch.setattr(A,'solve_ivp',fake)
    monkeypatch.setattr(V,'solve_ivp',fake)
    result=estimator_variants(sim,make_law(s),s.n,s.R,s.horizon)
    for k in ('d1','d1u','cres'):
        assert result[k+'_status'].startswith('failed')
        assert np.isnan(result[k+'_peak_m'])

@pytest.mark.parametrize('failure',['failed','truncated','coverage_gap'])
def test_invalid_cw_is_referred(case,failure):
    s,sim=case
    if failure=='failed': sim.status='failed'
    if failure=='truncated': sim.t_final=s.horizon/2
    if failure=='coverage_gap': sim.segments[0].t0=1.
    result=estimator_variants(sim,make_law(s),s.n,s.R,s.horizon)
    assert all(np.isnan(result[k+'_peak_m']) for k in ('d1','d1u','cres'))

def test_duplicate_switches_and_endpoints_do_not_change_feature(case):
    s,sim=case
    law=make_law(s)
    before=A.first_order_discrepancy(sim,law,s.n,s.R,s.horizon)['d1_peak_m']
    sim.sat_switch_times=list(reversed(sim.sat_switch_times))*2+[0.,s.horizon]
    after=A.first_order_discrepancy(sim,law,s.n,s.R,s.horizon)['d1_peak_m']
    assert after==before

def test_failure_after_completed_segment_has_no_partial_metrics(case,monkeypatch):
    s,sim=case
    sim.sat_switch_times=[s.horizon/2]
    calls=[]
    def fake(rhs,span,y0,**kw):
        a,b=span
        calls.append(span)
        def dense(t):
            return np.ones((6,len(np.atleast_1d(t))))
        dense.t_min=a; dense.t_max=b
        return SimpleNamespace(success=len(calls)==1,sol=dense,t=np.array([a,b]),
                               y=np.ones((6,2)),nfev=7,status=0)
    monkeypatch.setattr(V,'solve_ivp',fake)
    result=V._integrate(lambda t,e:e,sim,s.horizon)
    assert len(calls)==2 and result['nfev']==14
    assert result['status']=='failed' and 'segment' in result['failure_reason']
    assert all(np.isnan(result[k]) for k in ('peak_m','t_peak_s','final_m','peak_vel_mps'))

def test_nonfinite_dense_samples_include_failure_reason(case,monkeypatch):
    s,sim=case
    def fake(rhs,span,y0,**kw):
        a,b=span
        def dense(t):
            return np.full((6,len(np.atleast_1d(t))),np.nan)
        dense.t_min=a; dense.t_max=b
        return SimpleNamespace(success=True,sol=dense,t=np.array([a,b]),
                               y=np.ones((6,2)),nfev=5,status=0)
    monkeypatch.setattr(V,'solve_ivp',fake)
    result=V.estimator_variants(sim,make_law(s),s.n,s.R,s.horizon)
    for key in ('d1u','cres'):
        assert result[key+'_status']=='failed'
        assert 'non-finite' in result[key+'_failure_reason']
        assert np.isnan(result[key+'_final_m'])
