#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import numpy as np
from gridData import Grid
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase

from . import utils


AVOGADRO_CONSTANT_NA = 6.02214179e+23
BOLTZMANN_CONSTANT_KB = 0.0019872041


class Analysis(AnalysisBase):

    def __init__(self, atomgroup, gridsize=1., center=None, dimensions=None, **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)
        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize

        if center is None:
            receptor = self._u.select_atoms("protein or nucleic")
            self._center = np.mean(receptor.positions, axis=0)
        else:
            self._center = center

        if dimensions is None:
            self._dimensions = self._u.trajectory.ts.dimensions[:3]
        else:
            self._dimensions = dimensions

    def _prepare(self):
        self._positions = []

    def _single_frame(self):
        self._positions.extend(self._ag.atoms.positions)

    def _conclude(self):
        self._positions = np.array(self._positions)

        x, y, z = self._center
        sd = self._dimensions / 2.
        hrange = ((x - sd[0], x + sd[0]), (y - sd[1], y + sd[1]), (z - sd[2], z + sd[2]))
        hbins = np.round(self._dimensions / self._gridsize).astype(np.int)

        hist, edges = np.histogramdd(self._positions, bins=hbins, range=hrange)

        self.histogram = Grid(hist, origin=(edges[0][0], edges[1][0], edges[2][0]), delta=self._gridsize)
        self.density = Grid((hist - np.mean(hist)) / np.std(hist), origin=(edges[0][0], edges[1][0], edges[2][0]), delta=self._gridsize)

    def grid_free_energy(self, volume, concentration, temperature=300.):
        print("Warning: The GFE functionnality is very experimental.")
        n_voxel = np.prod(self.histogram.grid.shape)
        # Avoid 0 in the histogram
        hist = self.histogram + 1E-10

        N_o = ((1E-27 * volume) * AVOGADRO_CONSTANT_NA * concentration) / (n_voxel)
        gfe = -BOLTZMANN_CONSTANT_KB * temperature * np.log(hist.grid / N_o)

        # Empirical trick to remove the noise
        avg_value = np.mean(gfe[np.where(gfe < 0.)])
        gfe += np.abs(avg_value)
        gfe[gfe > 0.] = 0

        self.gfe = Grid(gfe, origin=self.histogram.origin, delta=self._gridsize)
