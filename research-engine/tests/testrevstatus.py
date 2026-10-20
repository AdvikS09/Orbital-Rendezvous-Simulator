from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from cwvalid.experiment import RunCfg, makelaw, runbranch
from cwvalid.scenario import axisscen
from cwvalid.estimate2 import estvar
import cwvalid.estimate as A
import cwvalid.estimate2 as V

@pytest.fixture
def case():
    s=axisscen('status-test','+R',3000.,6.,.05)
    sims,_=runbranch(s,RunCfg(models=('cw',),reference='cw',crosscheck=None))
    return s,sims['cw']

@pytest.mark.parametrize('failure',['solver','early_success','nonfinite','dense_gap'])
def testbadsegnever(case,monkeypatch,failure):
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
    res=estvar(sim,makelaw(s),s.n,s.R,s.horizon)
    for k in ('d1','d1u','cres'):
        assert res[k+'_status'].startswith('failed')
        assert np.isnan(res[k+'_peak_m'])

@pytest.mark.parametrize('failure',['failed','truncated','coverage_gap'])
def testbadcwref(case,failure):
    s,sim=case
    if failure=='failed': sim.status='failed'
    if failure=='truncated': sim.tfinal=s.horizon/2
    if failure=='coverage_gap': sim.segments[0].t0=1.
    res=estvar(sim,makelaw(s),s.n,s.R,s.horizon)
    assert all(np.isnan(res[k+'_peak_m']) for k in ('d1','d1u','cres'))

def testdupswitchends(case):
    s,sim=case
    law=makelaw(s)
    before=A.firstorderdisc(sim,law,s.n,s.R,s.horizon)['d1_peak_m']
    sim.satswitchtimes=list(reversed(sim.satswitchtimes))*2+[0.,s.horizon]
    after=A.firstorderdisc(sim,law,s.n,s.R,s.horizon)['d1_peak_m']
    assert after==before

def testfailcomplseg(case,monkeypatch):
    s,sim=case
    sim.satswitchtimes=[s.horizon/2]
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
    res=V.integr(lambda t,e:e,sim,s.horizon)
    assert len(calls)==2 and res['nfev']==14
    assert res['status']=='failed' and 'segment' in res['failure_reason']
    assert all(np.isnan(res[k]) for k in ('peak_m','t_peak_s','final_m','peak_vel_mps'))

def testnonfindensesamps(case,monkeypatch):
    s,sim=case
    def fake(rhs,span,y0,**kw):
        a,b=span
        def dense(t):
            return np.full((6,len(np.atleast_1d(t))),np.nan)
        dense.t_min=a; dense.t_max=b
        return SimpleNamespace(success=True,sol=dense,t=np.array([a,b]),
                               y=np.ones((6,2)),nfev=5,status=0)
    monkeypatch.setattr(V,'solve_ivp',fake)
    res=V.estvar(sim,makelaw(s),s.n,s.R,s.horizon)
    for key in ('d1u','cres'):
        assert res[key+'_status']=='failed'
        assert 'non-finite' in res[key+'_failure_reason']
        assert np.isnan(res[key+'_final_m'])
