"""Pruebas científicas independientes para los cambios del Falso Observatorio."""
import itertools
import numpy as np
import pytest
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity_batch, estimate_rm_unwrapped_lsq
from examples.falso_observatorio.estudio_parametrico import wilson_interval, threshold_crossing, estres_estadistico

W = 299792458.0/(np.array([5., 2., 1.4])*1e9)


def brute_force(w, p, lo, hi, nrange):
    # Oráculo independiente: producto completo de enteros, sin interpolación.
    x = w**2
    xc = x-x.mean()
    best = np.inf
    for ns in itertools.product(nrange, repeat=len(w)-1):
        y = p+np.pi*np.array([0, *ns])
        rm = np.clip(np.dot(xc, y)/np.dot(xc, xc), lo, hi)
        intercept = (y-rm*x).mean()
        cost = np.sum((y-rm*x-intercept)**2)
        best = min(best, cost)
    return best


@pytest.mark.parametrize("m", [3, 4])
def test_costo_global_coincide_con_enumeracion_completa(m):
    w = W if m == 3 else np.r_[W, 299792458/(3e9)]
    rng = np.random.default_rng(304+m)
    psi = rng.uniform(-np.pi/2, np.pi/2, (m, 6))
    fit = resolve_n_pi_ambiguity_batch(w, psi, -300, 400, return_diagnostics=True)
    for j in range(6):
        expected = brute_force(w, psi[:,j], -300, 400, range(-9,10))
        assert fit["cost"][j] == pytest.approx(expected, abs=1e-11)
    assert np.all((fit["rm"] >= -300) & (fit["rm"] <= 400))
    np.testing.assert_allclose(fit["psi_unwrapped"], psi+np.pi*fit["n_pi"], atol=2e-14)


def test_invariante_al_orden_de_canales_y_escalar():
    truth = np.array([-999.2, -50., 293.5, 1000.])
    psi = (0.5+W[:,None]**2*truth+np.pi/2)%np.pi-np.pi/2
    a = resolve_n_pi_ambiguity_batch(W, psi, -1500,1500)
    np.testing.assert_allclose(a, truth, atol=1e-10)
    for order in itertools.permutations(range(3)):
        b = resolve_n_pi_ambiguity_batch(W[list(order)], psi[list(order)], -1500,1500)
        np.testing.assert_allclose(a,b,atol=1e-10)
    for j in range(len(truth)):
        assert estimate_rm_unwrapped_lsq(W,psi[:,j],-1500,1500) == pytest.approx(a[j])


def test_dos_canales_no_identifican_la_rama_aun_con_ruido():
    rng = np.random.default_rng(56)
    psi = rng.uniform(-np.pi/2,np.pi/2,(2,100))
    fit = resolve_n_pi_ambiguity_batch(W[:2],psi,-1500,1500,return_diagnostics=True)
    assert np.all(~fit["unique"])
    assert np.max(fit["cost"]) < 1e-20


def test_dispersion_local_concuerda_con_propagacion_analitica():
    rng = np.random.default_rng(815)
    sigma = .005
    n = 10000
    phase = .5+250*W[:,None]**2
    q = np.cos(2*phase)+rng.normal(0,sigma,(3,n))
    u = np.sin(2*phase)+rng.normal(0,sigma,(3,n))
    psi = .5*np.arctan2(u,q)
    fitted = resolve_n_pi_ambiguity_batch(W,psi,-1500,1500)
    predicted = sigma/(2*np.sqrt(np.sum((W**2-(W**2).mean())**2)))
    assert np.std(fitted-250) == pytest.approx(predicted, rel=.06)
    assert abs(np.mean(fitted-250)) < .005


@pytest.mark.parametrize("w,p,lo,hi", [
    (W, np.zeros((3,1)), 0,0),
    (W, np.zeros((3,1)), np.nan,100),
    ([.1,.1],np.zeros((2,1)), -10,10),
    (W,np.full((3,1),np.nan),-100,100),
    (W,np.zeros((3,0)),-100,100),
    ([0.,.2],np.zeros((2,1)),-10,10),
])
def test_rechaza_entradas_no_validas(w,p,lo,hi):
    with pytest.raises(ValueError):
        resolve_n_pi_ambiguity_batch(w,p,lo,hi)


def test_wilson_extremos_y_cruce_no_alcanzado():
    lo, hi = wilson_interval(np.array([0,100]),100)
    assert lo[0] == pytest.approx(0,abs=1e-15)
    assert hi[0] > .03
    assert lo[1] < .97
    assert hi[1] == pytest.approx(1)
    assert threshold_crossing([0,.1],[0,.05])["estimate"] is None
    crossed = threshold_crossing([0,.1],[0,.2])
    assert crossed["estimate"] == pytest.approx(.05)
    assert crossed["grid_bracket"] == [0,.1]
    assert threshold_crossing([0,.1],[.5,.6])["status"] == "already_exceeded"


def test_reproducible_y_distingue_ambiguedad_del_error():
    kwargs = dict(sigmas=[0,.05],configuraciones={"3F":[0,1,2],"2F":[0,1]},n_muestra=120,n_repeticiones=2,seed=91)
    a = estres_estadistico(**kwargs)
    b = estres_estadistico(**kwargs)
    assert a == b
    assert a["3F"]["failure_counts"][0] == 0
    assert a["3F"]["ci95_high"][0] > 0
    assert a["2F"]["nonidentifiable_fraction"][0] == 1
    assert 0 < a["2F"]["error_fraction"][0] < 1
