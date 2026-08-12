#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Parse finished MMGBSA/decomposition runs and map them back onto the hotspot data
# model. Reuses AutoPath's own parsers (``autopath.ap_PLIP.ProteinLigandAnalyzer``)
# rather than re-implementing FINAL_RESULTS/FINAL_DECOMP parsing; autopath is imported
# lazily so cosolvkit does not hard-depend on it.
#
# THE resid TRAP: ``ante-MMPBSA.py`` strips solvent/ions/other-probe-copies (and, for
# decomposition runs, unsupported transition metals) FIRST and renumbers the survivors
# from 1. Every resid the decomposition table reports is in that STRIPPED-COMPLEX
# numbering, not the original topology's. ``stripped_index_to_resid`` inverts
# ``cosolvkit.cli.refine_hotspots_jobs.post_strip_ligand_index`` to undo this, and
# ``collect_decomposition`` cross-checks the decomposition's own resname against the
# topology's resname at the mapped resid so a wrong mapping fails loudly instead of
# silently annotating the wrong residue.
#

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Directory name written by `_build_mmgbsa_inputs`: f"{occ.probe_resname}{occ.probe_resid}".
PROBE_DIR_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

_STRIP_MASK_RE = re.compile(r'strip_mask\s*=\s*"([^"]*)"')
_ENDFRAME_RE = re.compile(r"^\s*endframe\s*=\s*(\d+)", re.MULTILINE)
_STARTFRAME_RE = re.compile(r"^\s*startframe\s*=\s*(\d+)", re.MULTILINE)
_INTERVAL_RE = re.compile(r"^\s*interval\s*=\s*(\d+)", re.MULTILINE)


def parse_results(results_path):
    """Thin wrapper over AutoPath's Differences-table parser.

    :param results_path: path to ``FINAL_RESULTS_mmpbsa.dat``.
    :return: ``{"delta_total", "std_dev", "std_err", "components"}`` where
        ``components`` is ``{component_name: {"average", "std_dev", "std_err"}}`` for
        every row of the Differences table, including ``"DELTA TOTAL"`` itself.
    :raises ValueError: if the table has no ``DELTA TOTAL`` row.
    """
    from autopath.ap_PLIP import ProteinLigandAnalyzer

    df = ProteinLigandAnalyzer.parse_mmpbsa_differences_table(results_path)
    components = {
        str(row.Component): {
            "average": float(row.Average),
            "std_dev": float(row.Std_Dev),
            "std_err": float(row.Std_Err_Mean),
        }
        for row in df.itertuples()
    }
    total = df[df["Component"] == "DELTA TOTAL"]
    if total.empty:
        raise ValueError(f"No 'DELTA TOTAL' row in the Differences table of {results_path}.")
    row = total.iloc[0]
    return {
        "delta_total": float(row["Average"]),
        "std_dev": float(row["Std_Dev"]),
        "std_err": float(row["Std_Err_Mean"]),
        "components": components,
    }


def parse_decomposition(decomp_path):
    """Thin wrapper over AutoPath's per-residue decomposition parser.

    :param decomp_path: path to ``FINAL_DECOMP_mmpbsa.dat``.
    :return: DataFrame with ``resname``, ``resid`` (STRIPPED-COMPLEX numbering —
        see module docstring), ``location`` (``'R'``/``'L'``), one
        ``{group}_Avg/_StdDev/_StdErr`` column per energy group, and ``label``.
    """
    from autopath.ap_PLIP import ProteinLigandAnalyzer

    return ProteinLigandAnalyzer.parse_mmpbsa_deltas_all_components(decomp_path)


def read_strip_mask(job_dir):
    """Read the ``strip_mask`` value recorded in ``<job_dir>/mmgbsa.in``.

    This is the mask actually handed to ``ante-MMPBSA.py`` for this job (verbatim, as
    written by ``refine_hotspots_jobs._build_mmgbsa_inputs``), so deriving the strip
    sets from it is guaranteed consistent with what the run actually did — no need to
    recompute cosolvent species or metal proximity at collection time.

    :raises FileNotFoundError: if ``mmgbsa.in`` is missing.
    :raises ValueError: if it has no ``strip_mask`` line.
    """
    path = os.path.join(job_dir, "mmgbsa.in")
    with open(path) as fh:
        text = fh.read()
    m = _STRIP_MASK_RE.search(text)
    if not m:
        raise ValueError(f"No 'strip_mask' found in {path}.")
    return m.group(1)


def parse_strip_mask(strip_mask):
    """Split an Amber strip mask into resnames and explicit excluded resids.

    ``strip_mask`` is built by ``_build_mmgbsa_inputs`` as
    ``SOLVENT_STRIP + amber_exclude_mask(...) [+ ":" + decomp-dropped metal names]``:
    a run of ``:``-separated tokens where most are residue NAMES (letters) and exactly
    one — the ``amber_exclude_mask`` output — is a comma/dash range of resids (digits).

    :param strip_mask: e.g. ``":POP:HOH:WAT:NA:CL:K:MG:279-288,290-315:MN"``.
    :return: ``(strip_resnames: set[str], excluded_resids: set[int])``.
    """
    strip_resnames = set()
    excluded_resids = set()
    for token in strip_mask.strip().strip(":").split(":"):
        if not token:
            continue
        if re.fullmatch(r"[\d,\-]+", token):
            for part in token.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    excluded_resids.update(range(int(a), int(b) + 1))
                else:
                    excluded_resids.add(int(part))
        else:
            strip_resnames.add(token)
    return strip_resnames, excluded_resids


def stripped_index_to_resid(universe, strip_resnames, excluded_resids):
    """Invert :func:`cosolvkit.cli.refine_hotspots_jobs.post_strip_ligand_index`.

    Builds the FULL map from stripped-complex 1-based index to original topology
    resid, by walking the same residues in the same order and skipping the same ones
    that function skips — so for any ``keep_resid``,
    ``stripped_index_to_resid(u, sr, er)[post_strip_ligand_index(u, sr, er, keep_resid)]
    == keep_resid``.

    :param universe: MDAnalysis Universe of the ORIGINAL (unstripped) topology.
    :param strip_resnames: residue names removed by the strip mask.
    :param excluded_resids: resids of the other stripped molecules (original numbering).
    :return: ``{stripped_index: original_resid}``.
    """
    mapping = {}
    index = 0
    for res in universe.residues:
        resid = int(res.resid)
        if res.resname in strip_resnames or resid in excluded_resids:
            continue
        index += 1
        mapping[index] = resid
    return mapping


def _n_frames(job_dir):
    """Authoritative frame count for one MMGBSA job, with provenance.

    ``frames.json`` (written by ``write_frame_trajectory``) is authoritative. Falls
    back to the ``endframe`` AutoPath rewrote into ``<sysname>_mmgbsa.in`` only when
    ``frames.json`` is missing.

    :return: ``(n_frames, source)`` where ``source`` is ``"frames.json"`` or the
        rewritten input file's basename.
    :raises FileNotFoundError: if neither is present.
    """
    frames_json = os.path.join(job_dir, "frames.json")
    if os.path.isfile(frames_json):
        with open(frames_json) as fh:
            manifest = json.load(fh)
        return int(manifest["n_frames"]), "frames.json"

    candidates = sorted(
        f for f in os.listdir(job_dir)
        if f.endswith("_mmgbsa.in") and f != "mmgbsa.in"
    )
    if not candidates:
        raise FileNotFoundError(
            f"Neither frames.json nor a rewritten *_mmgbsa.in found in {job_dir}; "
            f"cannot determine n_frames."
        )
    rewritten = candidates[0]
    with open(os.path.join(job_dir, rewritten)) as fh:
        text = fh.read()
    end_m = _ENDFRAME_RE.search(text)
    if not end_m:
        raise ValueError(f"No 'endframe' line in {os.path.join(job_dir, rewritten)}.")
    start_m = _STARTFRAME_RE.search(text)
    step_m = _INTERVAL_RE.search(text)
    start = int(start_m.group(1)) if start_m else 1
    step = int(step_m.group(1)) if step_m else 1
    n = (int(end_m.group(1)) - start) // step + 1
    return n, rewritten


def parse_probe_dirname(dirname):
    """Split a ``mmgbsa/<PROBE><resid>`` leaf directory name into resname/resid.

    :return: ``(probe_resname, probe_resid)``.
    :raises ValueError: if the name is not ``<letters><digits>``.
    """
    m = PROBE_DIR_RE.match(dirname)
    if not m:
        raise ValueError(
            f"{dirname!r} does not look like a '<PROBE><resid>' MMGBSA job directory."
        )
    return m.group(1), int(m.group(2))


def collect_job(job_dir, probe_resname, probe_resid, source_label):
    """Read one finished ``mmgbsa/<PROBE><resid>/`` job directory.

    :param job_dir: the job's directory (containing ``FINAL_RESULTS_mmpbsa.dat`` once
        MMPBSA has finished).
    :param probe_resname: the probe's residue name, ORIGINAL topology numbering.
    :param probe_resid: the probe's resid, ORIGINAL topology numbering — this is
        `occ.probe_resid`, already original, not the stripped-complex index.
    :param source_label: the simulation this molecule's frames came from (the
        representative record's, when several were pooled).
    :return: :class:`~cosolvkit.analysis.core.models.MmgbsaResult`, or ``None`` (with
        a log line) when the run has not produced results yet.
    """
    from cosolvkit.analysis.core.models import MmgbsaResult

    results_path = os.path.join(job_dir, "FINAL_RESULTS_mmpbsa.dat")
    if not os.path.isfile(results_path):
        logger.info(
            "%s: no FINAL_RESULTS_mmpbsa.dat yet; job has not finished, skipping.",
            job_dir,
        )
        return None

    parsed = parse_results(results_path)
    n_frames, n_frames_source = _n_frames(job_dir)
    logger.info("%s: n_frames=%d (from %s).", job_dir, n_frames, n_frames_source)

    return MmgbsaResult(
        probe_resname=probe_resname,
        probe_resid=int(probe_resid),
        source_label=source_label,
        delta_total=parsed["delta_total"],
        std_dev=parsed["std_dev"],
        std_err=parsed["std_err"],
        n_frames=n_frames,
        components=parsed["components"],
        results_path=os.path.abspath(results_path),
    )


def collect_decomposition(job_dir, universe, strip_resnames, excluded_resids):
    """Read and re-index one job's per-residue decomposition, if it exists.

    Maps every ``resid`` from stripped-complex numbering to the original topology via
    :func:`stripped_index_to_resid`, keeping the stripped index alongside it, and
    cross-checks the decomposition's own resname against the topology's resname at the
    mapped resid. A mismatch means the mapping is wrong for this job (e.g. the wrong
    strip mask was used to build it) — this raises rather than writing a silently
    wrong row.

    :param job_dir: the job's directory.
    :param universe: MDAnalysis Universe of the ORIGINAL (unstripped) topology.
    :param strip_resnames: as used to build this job's strip mask.
    :param excluded_resids: as used to build this job's strip mask.
    :return: the decomposition DataFrame with an added ``stripped_resid`` column and
        ``resid`` replaced by the mapped ORIGINAL resid, or ``None`` (with a log line)
        if ``FINAL_DECOMP_mmpbsa.dat`` does not exist.
    :raises ValueError: if a stripped index has no mapping, or a mapped resid's
        topology resname disagrees with the decomposition's.
    """
    decomp_path = os.path.join(job_dir, "FINAL_DECOMP_mmpbsa.dat")
    if not os.path.isfile(decomp_path):
        logger.info(
            "%s: no FINAL_DECOMP_mmpbsa.dat; skipping per-residue decomposition.",
            job_dir,
        )
        return None

    df = parse_decomposition(decomp_path).copy()
    mapping = stripped_index_to_resid(universe, strip_resnames, excluded_resids)
    topology_resname = {int(r.resid): r.resname for r in universe.residues}

    original_resids = []
    for stripped_resid in df["resid"]:
        original = mapping.get(int(stripped_resid))
        if original is None:
            raise ValueError(
                f"{decomp_path}: stripped index {stripped_resid} has no mapping back "
                f"to the original topology — the strip mask used to build the "
                f"mapping does not match the one this MMPBSA run actually used."
            )
        original_resids.append(original)

    df["stripped_resid"] = df["resid"].astype(int)
    df["resid"] = original_resids

    for decomp_resname, resid in zip(df["resname"], df["resid"]):
        topo_resname = topology_resname.get(int(resid))
        if topo_resname != decomp_resname:
            raise ValueError(
                f"{decomp_path}: decomposition resname {decomp_resname!r} at mapped "
                f"resid {resid} does not match the topology's {topo_resname!r} — the "
                f"stripped-index-to-resid mapping is wrong for this job."
            )

    return df
