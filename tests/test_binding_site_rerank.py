import numpy as np
import pandas as pd
import pytest
from cosolvkit.analysis.hotspot_dashboard import (
    _load_binding_sites_csv, rerank_binding_sites, DEFAULT_DASHBOARD_WEIGHTS,
)


def _bs_df():
    # Two binding sites. A is best on every feature; B worst.
    return pd.DataFrame([
        dict(site_id=1, rank=2, combined=0.0, cosolvents="BEN,IMI", n_cosolvents=2,
             probe_coverage=1.0, n_hotspots=2, member_hotspot_ids="1,2",
             centroid_x=0.0, centroid_y=0.0, centroid_z=0.0, agfe_min=-3.0,
             agfe_mean_top_pct=-2.5, volume=100.0, solidity=0.9, extent=0.5,
             axis_major_length=3.0, axis_minor_length=2.0, favorable_atomtypes="Car,HBA",
             n_chemotypes=2, residence=20.0, accessible_fraction=0.30),
        dict(site_id=2, rank=1, combined=0.0, cosolvents="BEN", n_cosolvents=1,
             probe_coverage=0.5, n_hotspots=1, member_hotspot_ids="3",
             centroid_x=5.0, centroid_y=5.0, centroid_z=5.0, agfe_min=-1.0,
             agfe_mean_top_pct=-0.8, volume=50.0, solidity=0.6, extent=0.4,
             axis_major_length=2.0, axis_minor_length=1.0, favorable_atomtypes="Car",
             n_chemotypes=1, residence=10.0, accessible_fraction=0.70),
    ])


def _only(**kw):
    """Weight set with every default zeroed except the named terms.

    Listing a few keys by hand is fragile: `score_binding_sites` merges a partial dict over the
    defaults, so any term the test forgot stays at its default weight and quietly contributes.
    That is exactly what broke these two tests when `accessible_fraction` gained a non-zero
    default.
    """
    from cosolvkit.analysis.core.scoring import DEFAULT_BINDING_SITE_WEIGHTS
    return {**{k: 0.0 for k in DEFAULT_BINDING_SITE_WEIGHTS}, **kw}


def test_default_weights_constant():
    """The dashboard must not keep its own copy of the weights; see
    tests/test_default_weight_coherence.py for why the previous copy was wrong."""
    from cosolvkit.analysis.core.scoring import DEFAULT_BINDING_SITE_WEIGHTS
    assert DEFAULT_DASHBOARD_WEIGHTS == DEFAULT_BINDING_SITE_WEIGHTS


def test_rerank_default_weights_are_shape_dominated():
    """One term, `shape`, now outweighs every other term a site can win COMBINED.

    This fixture used to pin the opposite: site 1 lost `shape` but won affinity (3.0) +
    probe_coverage (2.0) + accessible_fraction (2.0) = 7.0 and still ranked first. The
    2026-08-07 binding-site refit demoted affinity to 1.0 (its sign agreed in only 4 of 6
    leave-one-pocket-out folds) and zeroed probe_coverage (it shipped at +2.0 while fitting
    NEGATIVE in 5 of 6 folds), which leaves site 1 with 1.0 + 2.0 = 3.0 against site 2's
    shape alone at 3.5. The ordering flips.

    Pinned deliberately rather than quietly updated, because it is the sharpest statement of
    what the default model now is: **shape-dominated**, carrying 3.5 of the 6.5 total weight.
    A site that is the wrong SHAPE cannot be rescued by being deep and enclosed. If a future
    fit moves shape below 3.0, this test flips back and should be re-read, not re-flipped.
    """
    out = rerank_binding_sites(_bs_df(), DEFAULT_DASHBOARD_WEIGHTS)
    r = out.set_index("site_id")
    assert r.loc[2, "rank"] == 1
    assert r.loc[1, "rank"] == 2
    # site 1: affinity 1.0 + accessible_fraction 2.0 (probe_coverage now contributes 0.0)
    assert r.loc[1, "combined"] == pytest.approx(3.0, abs=1e-9)
    # site 2: shape 3.5, scored lower-is-better, and site 2 has the lower solidity
    assert r.loc[2, "combined"] == pytest.approx(3.5, abs=1e-9)
    # output is sorted by rank ascending
    assert list(out["rank"]) == [1, 2]


def test_rerank_negative_volume_weight_flips_toward_smaller():
    w = _only(volume=-1)
    out = rerank_binding_sites(_bs_df(), w).set_index("site_id")
    # smaller-volume site (2) preferred
    assert out.loc[2, "rank"] == 1 and out.loc[1, "rank"] == 2


def test_rerank_handles_blank_residence():
    df = _bs_df()
    df.loc[df["site_id"] == 1, "residence"] = np.nan  # missing kinetics
    w = _only(kinetics=1)
    out = rerank_binding_sites(df, w).set_index("site_id")
    # site 2 has the only finite residence -> flat minmax -> 1.0; site 1 (NaN) -> 0.0
    assert out.loc[2, "rank"] == 1


def test_load_missing_returns_empty(tmp_path):
    assert _load_binding_sites_csv(str(tmp_path)).empty


def test_load_reads_binding_sites_csv(tmp_path):
    _bs_df().to_csv(tmp_path / "binding_sites.csv", index=False)
    df = _load_binding_sites_csv(str(tmp_path))
    assert len(df) == 2 and "combined" in df.columns
