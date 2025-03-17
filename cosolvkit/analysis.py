#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import os
import sys
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator
from gridData import Grid
from MDAnalysis import Universe
from MDAnalysis.analysis import rdf, align
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rms import RMSF
import matplotlib.pyplot as plt
import pandas as pd
from pymol import cmd
from cosolvkit.cosolvent_system import CosolventMolecule


BOLTZMANN_CONSTANT_KB = 0.0019872041  # kcal/(mol*K)

def _normalization(data, a:float=None, b:float=None):
    """_summary_

    :param data: list of data points
    :type data: list
    :param a: int a, defaults to None
    :type a: int, optional
    :param b: int b, defaults to None
    :type b: int, optional
    :return: normalized data
    :rtype: list
    """
    min_data = np.min(data)
    max_data = np.max(data)
    epsilon = 1e-20  # small value to avoid division by zero
    return a + ((data - min_data) * (b - a)) / (max_data - min_data + epsilon)

def dynamic_threshold_smoothed(occupancy, sigma=1, percentile=85):
    """
    Smooths the occupancy map and sets a threshold dynamically.

    :param occupancy: 3D numpy array of occupancy values.
    :param sigma: Standard deviation for Gaussian smoothing.
    :param percentile: Percentile for thresholding.
    :return: Computed threshold.
    """
    occupancy = gaussian_filter(occupancy, sigma=sigma)
    return np.percentile(occupancy, percentile)

def compute_local_entropy(occupancy, radius=2):
    """
    Computes the local spatial entropy (LSE) of the occupancy map. Uses a very simple and cheap
    method to estimate the local entropy of each voxel based on a smoothed occupancy map.
    A Gaussian filter is applied to the occupancy map to estimate local probabilities, which are
    then used to compute the local entropy. A KDE-based method would be toooo slow.

    :param occupancy: 3D numpy array of occupancy values.
    :param radius: Defines the neighborhood size.
    :return: 3D numpy array of local entropy values.
    """
    
    # Normalize occupancy to get probability distribution
    occupancy_prob = occupancy / np.sum(occupancy)
    
    # Smooth the occupancy to estimate local probabilities
    smoothed = gaussian_filter(occupancy_prob, sigma=radius)
    
    # Compute entropy per voxel
    local_entropy = -smoothed * np.log(smoothed + 1e-10)  # Avoid log(0)
    
    return local_entropy

def entropy_corrected_free_energy(gfe, occupancy, lambda_factor=0.5, radius=2):
    """
    Applies entropy correction to the grid free energy map.

    :param gfe: 3D numpy array of free energy values.
    :param occupancy: 3D numpy array of occupancy values.
    :param lambda_factor: Scaling factor for entropy correction.
    :param radius: Neighborhood size for entropy calculation.
    :return: 3D numpy array of corrected free energy values.
            """
    # Compute local entropy
    local_entropy = compute_local_entropy(occupancy, radius=radius)
    
    # Apply entropy correction
    gfe_corrected = gfe + lambda_factor * local_entropy

    return gfe_corrected

def _grid_free_energy(hist, n_atoms, n_frames, temperature=300., entropy_correction=False):
    """
    Compute the atomic grid free energy (GFE) from a given histogram.
    
    :param hist: Histogram of cosolvent occupancy in each voxel
    :param n_atoms: Total number of cosolvent atoms (not total system atoms)
    :param n_frames: Number of frames in the trajectory
    :param temperature: Temperature in Kelvin (default 300K)
    :param entropy_correction: Apply entropy correction to the free energy map (default False)
    :return: 3D numpy array of free energy values (same shape as `hist`)
    """
        
    # Before the volume_water was calculated as the total volume of the box, but it should be the volume of the solvent region
    # this approximation is not correct, but it is a good starting point
    n_accessible_voxels = np.sum(hist > 0)  # Count nonzero occupancy voxels

    # Apply occupancy filtering: remove low-occupancy grid points
    # occupancy_threshold = dynamic_threshold_smoothed(occupancy, sigma=1, percentile=85)s
    # hist[occupancy < occupancy_threshold] = 0

    N_o = n_atoms / n_accessible_voxels  # Bulk probability of cosolvent
    N = hist / n_frames  # Local probability in the grid

    # print(f"Min N: {np.min(N)}, Max N: {np.max(N)}, Min N_o: {np.min(N_o)}, Max N_o: {np.max(N_o)}")

    #if hist contains very low values (or zeros), N = hist / n_frames can be much smaller than N_o
    # making log(N / N_o) too negative and gfe extremely large.
    N = np.maximum(N, 1E-10)
   
    gfe = -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)

    print(f'Min GFE: {np.min(gfe)}, Max GFE: {np.max(gfe)}')

    if entropy_correction:
        gfe = entropy_corrected_free_energy(gfe, hist, lambda_factor=500, radius=2)
        print(f'Min GFE corrected: {np.min(gfe)}, Max GFE corrected: {np.max(gfe)}')
    
    return gfe

def _smooth_grid_free_energy(gfe, energy_cutoff: float = 0, 
                             sigma: float = 1, 
                            ):
    """
    Smooths and filters the grid free energy (GFE) map.

    :param gfe: 3D numpy array of grid free energy values.
    :param energy_cutoff: Cutoff energy (default: .0 kcal/mol). Only values below this are retained.
    :param sigma: Standard deviation for Gaussian smoothing (default: 1).
    :return: Smoothed and filtered grid free energy map (new array).
    """

    gfe_filtered = np.copy(gfe)

    # the energy cutoff is applied before smoothing, 
    # gfe_filtered[gfe_filtered >= energy_cutoff] = 0.0

    # Apply Gaussian smoothing AFTER filtering (not sure if this is the best approach)
    gfe_smoothed = gaussian_filter(gfe_filtered, sigma=sigma)
    
    # print(f'Energy cutoff is: {energy_cutoff}')

    # Keep only favorable energy values after smoothing
    gfe_smoothed[gfe_smoothed >= energy_cutoff] = 0.0

    print(f'Min gfe_smoothed: {np.min(gfe_smoothed)}, Max gfe_smoothed: {np.max(gfe_smoothed)}')

    # Normalization has not no effect
    # gfe_smooth_norm = _normalization(gfe_smoothed,np.min(gfe_smoothed), 0.0)

    # print(f'Min gfe_smooth_norm: {np.min(gfe_smoothed)}, Max gfe_smooth_norm: {np.max(gfe_smoothed)}')

    return gfe_smoothed

def _grid_density(hist):
    return (hist - np.mean(hist)) / np.std(hist)

def _subset_grid(grid, center, box_size, gridsize=0.5):

    #FIXME I think this part of the code is never triggered, not sure if we need this

    # Create grid interpolator
    # Number of midpoints is equal to the number of grid points
    grid_interpn = RegularGridInterpolator(grid.midpoints, grid.grid)

    # Create sub grid coordinates
    # We get first the edges of the grid box, and after the midpoints
    # So this we are sure (I guess) that the sub grid is well centered on center
    # There might be a better way of doing this... Actually I tried, but didn't worked very well.
    x, y, z = center
    sd = box_size / 2.
    hbins = np.round(box_size / gridsize).astype(int)
    edges = (np.linspace(0, box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
             np.linspace(0, box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
             np.linspace(0, box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
    midpoints = (edges[0][:-1] + np.diff(edges[0]) / 2.,
                 edges[1][:-1] + np.diff(edges[1]) / 2.,
                 edges[2][:-1] + np.diff(edges[2]) / 2.)
    X, Y, Z = np.meshgrid(midpoints[0], midpoints[1], midpoints[2])
    xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    # Configuration of the sub grid
    origin_subgrid = (midpoints[0][0], midpoints[1][0], midpoints[2][0])
    shape_subgrid = (midpoints[0].shape[0], midpoints[1].shape[0], midpoints[2].shape[0])

    # Do interpolation
    sub_grid_values = grid_interpn(xyzs)
    sub_grid_values = sub_grid_values.reshape(shape_subgrid)
    sub_grid_values = np.swapaxes(sub_grid_values, 0, 1)
    sub_grid = Grid(sub_grid_values, origin=origin_subgrid, delta=gridsize)

    return sub_grid

def _export(fname, grid, gridsize=0.5, center=None, box_size=None):
    assert (center is None and box_size is None) or (center is not None and box_size is not None), \
           "Both center and box size have to be defined, or none of them."

    if center is None and box_size is None:
        grid.export(fname)
    elif center is not None and box_size is not None:
        center = np.array(center)
        box_size = np.array(box_size)

        assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
        assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
        assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

        sub_grid = _subset_grid(grid, center, box_size, gridsize)
        sub_grid.export(fname)


class Analysis(AnalysisBase):
    """Analysis class to generate density grids

    :param AnalysisBase: Base MDAnalysis class
    :type AnalysisBase: AnalysisBase
    """
    def __init__(self, atomgroup, gridsize=0.5, **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)

        if atomgroup.n_atoms == 0:
            print("Error: no atoms were selected.")
            sys.exit(1)

        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize
        self._nframes = 0
        self._n_atoms = atomgroup.n_atoms
        self._center = None
        self._box_size = None

    def _prepare(self):
        self._positions = []
        self._centers = []
        self._dimensions = []

    def _single_frame(self):
        self._positions.append(self._ag.atoms.positions.astype(float))
        self._dimensions.append(self._u.dimensions[:3])
        self._centers.append(self._u.atoms.center_of_geometry())
        self._nframes += 1

    def _conclude(self):
        self._positions = np.array(self._positions, dtype=float)
        self._box_size = np.mean(self._dimensions, axis=0)
        self._center = np.mean(self._centers, axis=0)

        # Get all the positions
        positions = self._get_positions()

        # Get grid edges and origin
        x, y, z = self._center
        sd = self._box_size / 2.
        hbins = np.round(self._box_size / self._gridsize).astype(int)
        self._edges = (np.linspace(0, self._box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
                       np.linspace(0, self._box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
                       np.linspace(0, self._box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
        origin = (self._edges[0][0], self._edges[1][0], self._edges[2][0])

        hist, edges = np.histogramdd(positions, bins=self._edges)
        self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
        self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop,:,:]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)
        return positions

    def atomic_grid_free_energy(self, temperature=300., atom_radius=1.4, smoothing=True):
        """Compute grid free energy.
        """
        agfe = _grid_free_energy(self._histogram.grid, self._n_atoms, self._nframes, temperature)

        if smoothing:
            # We divide by 2 in order to have radius == 2 sigma
            agfe = _smooth_grid_free_energy(agfe, sigma=atom_radius / 2., energy_cutoff=0)

        self._agfe = Grid(agfe, edges=self._histogram.edges)

    def export_histogram(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export histogram maps
        """
        _export(fname, self._histogram, gridsize, center, box_size)

    def export_density(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps
        """
        _export(fname, self._density, gridsize, center, box_size)

    def export_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export atomic grid free energy
        """
        _export(fname, self._agfe, gridsize, center, box_size)

class Report:
    """Report class. This is the main class that takes care of post MD simulation processing and analysis.
    """
    def __init__(self, log_file, traj_file, top_file, cosolvents_path, out_path):
        """_summary_

        :param log_file: log file generated by MD Simulation. In CosolvKit this is called statistics.csv
        :type log_file: str
        :param traj_file: Trajectory file generated by MD Simulation.
        :type traj_file: str
        :param top_file: Topology file generated by CosolvKit.
        :type top_file: str
        :param cosolvents_path: path to the json file defining the cosolvents present in the system.
        :type cosolvents_path: str
        :param out_path: path to where to save the results.
        :type out_path: str
        """
        self.statistics = log_file
        self.trajectory = traj_file
        self.topology = top_file
        self.universe = Universe(self.topology, self.trajectory)

        self.cosolvents = list()

        with open(cosolvents_path) as fi:
            cosolvents_d = json.load(fi)
        for cosolvent in cosolvents_d:
            self.cosolvents.append(CosolventMolecule(**cosolvent))

        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        # this will be checked when creating the sessions
        self.avg_pdb_path = os.path.join(self.out_path, "averaged_trajectory.pdb")

        self._volume = None
        self._temperature = None
        self._potential_energy = None
        self._potential_energy, self._temperature, self._volume = self._get_temp_vol_pot(self.statistics)
        
        return
    
    def _plot_rmsf(self, rmsf_df):
        """Plots the RMSF of the protein residues.

        :param rmsf_df: dataframe with the RMSF data per atom.
        :type rmsf_df: pd.DataFrame
        """
        # Group by residue and calculate the mean RMSF
        rmsf_df = rmsf_df.groupby('residue').mean().reset_index()

        fig, ax = plt.subplots()
        ax.plot(rmsf_df['residue'], rmsf_df['RMSF'])
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSF')
        ax.set_title('RMSF of the protein residues')
        plt.savefig(os.path.join(self.out_path, "rmsf_by_residue.png"))
        return

    def _rmsf_analysis(self, avg_selection):
        """Computes the RMSF of the protein residues. 
        The funciton also generates the average structure of the trajectory and colors the residues by RMSF.
        This conformtion will be used as a reference for the pymol session.
        As for the density analysis, this function also asumes that the trajectory is already aligned.
        :param avg_selection: selection string to average the trajectory.
        :type avg_selection: str
        """
        average = align.AverageStructure(self.universe, None,
                                        select=avg_selection,
                                        ).run()
        
        u_avg = average.results.universe
        aligner = align.AlignTraj(self.universe, u_avg, 
                                  select='protein and name CA', in_memory=True).run()

        selection = self.universe.select_atoms('protein')
        residues = selection.resids
        rmsf = RMSF(selection).run()

        rmsf_df = pd.DataFrame({'residue': residues, 'RMSF': rmsf.results.rmsf})
        rmsf_df.index.name = 'atom'

        self.universe.add_TopologyAttr('tempfactors') # add empty attribute for all atoms
        for residue, r_value in zip(selection.residues, rmsf.results.rmsf):
            residue.atoms.tempfactors = r_value
        
        selection.write(self.avg_pdb_path)
        rmsf_df.to_csv(os.path.join(self.out_path, "rmsf_by_atom.csv"))    
        self._plot_rmsf(rmsf_df)

        return

    def generate_report(self):
        """Creates the main plots for RDFs, autocorrelations and equilibration.
        """
        print("Generating report...")
        rdf_path = os.path.join(self.out_path, "RDFs")
        os.makedirs(rdf_path, exist_ok=True)

        print('Plotting equilibration data')
        self._plot_temp_vol_pot(self.out_path)

        print('Plotting RMSF data')
        self._rmsf_analysis(avg_selection='protein')

        print("Plotting RDFs")
        self._rdf_mda(self.universe, self.cosolvents, rdf_path)

        return
    
    def generate_density_maps(self, temperature:float=None, analysis_selection_string=""):
        """Generates the desnity maps for the target cosolvents.

        :param analysis_selection_string: MD Analysis selection string if want to generate densities only for specific molecules, defaults to ""
        :type analysis_selection_string: str, optional
        """
        print("Generating density maps...")

        if temperature is None: # If temperature is not passed, so we take the last one from statistics
            temperature = self._temperature[-1]

        if analysis_selection_string == "":
            print("No cosolvent specified for the densities analysis. Generating a density map for each cosolvent.")
            for cosolvent in self.cosolvents:
                selection_string = f"resname {cosolvent.resname}"
                self._run_analysis(selection_string=selection_string,
                                   temperature=temperature,
                                   cosolvent_name=cosolvent.resname)
        else:
            print(f"Generating density maps for the following selection string: {analysis_selection_string}")
            self._run_analysis(selection_string=analysis_selection_string, 
                               temperature=temperature,
                               cosolvent_name=None)
        return
    
    def generate_pymol_reports(self, density_files:list[str]=None, 
                               selection_string:str=None, 
                               reference_pdb:str=None):
        """Generate the PyMol reports from the density maps. The average structure is always used as a reference. 
        You can also include a reference pdb file and specify the residues of interest. 

        :param reference_pdb: reference pdb file to load in PyMol.
        :type reference_pdb: str
        :param density_files: list of density files to include in the same PyMol session. Limited to 5.
        :type density_files: list
        :param selection_string: PyMol selection string if willing to specify target residues.
        :type selection_string: str
        """
        colors = ['marine', 
                  'orange', 
                  'magenta',
                  'salmon',
                  'purple']

        assert len(density_files) <= len(colors), "Error! Too many density files, not enough colors available!"
        
        if not os.path.exists(self.avg_pdb_path):
            # if the average pdb was not generated in the report, we generate it here
            self._rmsf_analysis(avg_selection='protein')

        structures = {'average_structure': self.avg_pdb_path}
        if reference_pdb is not None and reference_pdb.endswith('.pdb'):
            reference_pdb_name = os.path.basename(reference_pdb).split('.')[0]
            structures[reference_pdb_name] = reference_pdb

        cmd_string = ""

        for structure_name, pdb_path in structures.items():

            # Load topology and first frame of the trajectory
            cmd.load(pdb_path, structure_name)
            cmd_string += f"cmd.load('{pdb_path}', {structure_name})\n"

            # Set structure's color
            cmd.color("grey50", f"{structure_name} and name C*")
            cmd_string += f"cmd.color('grey50', '{structure_name} and name C*')\n"

        for color, density in zip(colors, density_files):
            dens_name = os.path.basename(density).split('.')[0]
            # print(f"Loading density map: {dens_name}")

            dx_data = Grid(density)
            # calculate 0.001 quantile. This works for agfe maps
            dx_01 = np.quantile(dx_data.grid, 0.001)
            # print(f"0.1% of the density map is: {dx_01}")

            cmd.load(density, f'{dens_name}_map')
            cmd_string += f"cmd.load('{density}', '{dens_name}_map')\n"

            # Create isomesh for hydrogen bond probes
            cmd.isomesh(f'{dens_name}_mesh', f'{dens_name}_map', dx_01)
            cmd_string += f"cmd.isomesh('{dens_name}_mesh', '{dens_name}_map', {dx_01})\n"

            # Color the hydrogen bond isomesh
            cmd.color(color, f'{dens_name}_mesh')
            cmd_string += f"cmd.color('{colors}', '{dens_name}_mesh')\n"
            
        # Show sticks for the residues of interest
        if selection_string != '':
            cmd.show("sticks", selection_string)
            cmd_string += f"cmd.show('sticks', '{selection_string}')\n"

        cmd.hide("spheres")
        # Set valence to 0 - no double bonds
        # cmd.set("valence", 0)
        cmd.set('specular', 1)
        # Set cartoon_side_chain_helper to 1 - less messy
        cmd.set("cartoon_side_chain_helper", 1)
        # Set background color
        cmd.bg_color("white") #grey80

        cmd_string += "cmd.hide('spheres')\n"
        # cmd_string += "cmd.set('valence', 0)\n"
        cmd_string += "cmd.set('specular', 1)\n"
        cmd_string += "cmd.set('cartoon_side_chain_helper', 1)\n"
        cmd_string += "cmd.bg_color('white')"
        
        with open(os.path.join(self.out_path, "pymol_session_cmd.pml"), "w") as fo:
            fo.write(cmd_string)
            
        cmd.save(os.path.join(self.out_path, "pymol_results_session.pse"))
        return

    def _run_analysis(self, selection_string, temperature, cosolvent_name=None):
        """Creates Analysis object and generates densities.

        :param selection_string: MD Analysis selection string.
        :type selection_string: str
        :param temperature: temperature of the system.
        :type temperature: float
        :param cosolvent_name: name of the cosolvent if not analysing all the cosolvents in the system, defaults to None
        :type cosolvent_name: str, optional
        """
        fig_density_name = os.path.join(self.out_path, f"map_density.dx")
        fig_energy_name =  os.path.join(self.out_path, f"map_agfe.dx")
        if cosolvent_name is not None:
            fig_density_name = os.path.join(self.out_path, f"map_density_{cosolvent_name}.dx")
            fig_energy_name =  os.path.join(self.out_path, f"map_agfe_{cosolvent_name}.dx")
        analysis = Analysis(self.universe.select_atoms(selection_string), verbose=True)
        analysis.run()
        analysis.atomic_grid_free_energy(temperature)
        analysis.export_density(fig_density_name)
        analysis.export_atomic_grid_free_energy(fig_energy_name)
        self.density_file = fig_density_name
        return

    def _get_temp_vol_pot(self, log_file):
        """Returns temperature, volume and potential energy of the system during the MD simulation.

        :param log_file: log file generated by the MD simulation. In CosolvKit is statistics.csv.
        :type log_file: str
        :return: potential energy, temperature and volume of the system for each frame.
        :rtype: tuple(list, list, list)
        """
        df = pd.read_csv(log_file)
        pot_e = list(df["Potential Energy (kJ/mole)"])
        temp = list(df["Temperature (K)"])
        vol = list(df["Box Volume (nm^3)"])
        return pot_e, temp, vol

    def _plot_temp_vol_pot(self, outpath=None):
        """Plots equilibration data.

        :param outpath: path to where to save the plot, defaults to None
        :type outpath: str, optional
        """
        if outpath is not None:
            fig_name = f"{outpath}/simulation_statistics.png"

        fig, axs = plt.subplots(3, 1, figsize=(12, 6))

        axs[0].plot(self._potential_energy, color='green', linewidth=2)
        axs[0].set_title('Potential Energy',)
        axs[0].set_xlabel('Frames')
        axs[0].set_ylabel('Energy (kJ/mole)')
    
        axs[1].plot(self._volume, color='blue', linewidth=2)
        axs[1].set_title('Volume')
        axs[1].set_xlabel('Frames')
        axs[1].set_ylabel('Volume (nm^3)')

        axs[2].plot(self._temperature, color='red', linewidth=2)
        axs[2].set_title('Temperature')
        axs[2].set_xlabel('Frames')
        axs[2].set_ylabel('Temperature (K)')

        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

        return 
    
    def _rdf_mda(self, universe: Universe, cosolvents: list, outpath=None):
        """Generates the plots for RDFs and Autocorrelations.

        :param universe: MD Analysis Universe that is created from the topology and trajectories.
        :type universe: Universe
        :param cosolvents: list of cosolvents in the system
        :type cosolvents: list
        :param outpath: path to where to save the plots, defaults to None
        :type outpath: str, optional
        """
        np.seterr(divide='ignore', invalid='ignore')
        wat_resname = "HOH"
        # if top.endswith("cosolv_system.prmtop"):
        #     wat_resname = "WAT"
        oxygen_atoms = universe.select_atoms(f"resname {wat_resname} and name O")
        sim_frames = len(universe.trajectory)
        step_size = int(sim_frames/250)
        if step_size < 1:
            step_size = 1
        n_bins = 150
        for cosolvent in cosolvents:
            cosolvent_name = cosolvent.resname
            r_max = 15
                
            cosolvent_residues = universe.select_atoms(f'resname {cosolvent_name}')
            atoms_names = cosolvent_residues.residues[0].atoms.names
            for cosolvent_atom in set(atoms_names):
                max_y = 0
                if "H" in cosolvent_atom: continue
                print(f"Analysing {cosolvent_name}-{cosolvent_atom}")
                fig, ax = plt.subplots(2, 2, sharex=False, sharey=False)
                plt.tight_layout(pad=3.0)
                # Here compute RDF between same atoms and different molecules
                atoms = cosolvent_residues.select_atoms(f'name {cosolvent_atom}')
                irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
                irdf.run(start=0, step=step_size)
                max_y = max(irdf.results.rdf)
                ax[0][0].plot(irdf.results.bins, irdf.results.rdf, label="RDF")
                ax[0][0].set_xlabel(r'$r$ $\AA$')
                ax[0][0].set_ylabel("$g(r)$")
                ax[0][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom}")
                # ax[0][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom} every {step_size} frames")
                leg = ax[0][0].legend(handlelength=0, handletextpad=0, fancybox=True)
                for item in leg.legendHandles:
                    item.set_visible(False)
                
                ax[1][0] = self._plot_autocorrelation(data=irdf.results.rdf,
                                                        ax=ax[1][0], 
                                            cosolvent_name1=cosolvent_name, 
                                            cosolvent_atom1=cosolvent_atom, 
                                            cosolvent_name2=cosolvent_name, 
                                            cosolvent_atom2=cosolvent_atom)
                # Here compute RDF between atom and water's oxygen
                irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
                irdf.run(start=0, step=step_size)
                max_y = max(irdf.results.rdf)
                irdf.run()
                ax[0][1].plot(irdf.results.bins, irdf.results.rdf, label="RDF")
                ax[0][1].set_xlabel(r'$r$ $\AA$')
                ax[0][1].set_ylabel("$g(r)$")
                ax[0][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O")
                # ax[0][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O every {step_size} frames")
                leg = ax[0][1].legend(handlelength=0, handletextpad=0, fancybox=True)
                for item in leg.legendHandles:
                    item.set_visible(False)

                self._plot_autocorrelation(data=irdf.results.rdf, 
                                             ax=ax[1][1], 
                                             cosolvent_name1=cosolvent_name, 
                                             cosolvent_atom1=cosolvent_atom, 
                                             cosolvent_name2="HOH", 
                                             cosolvent_atom2="O")
                if outpath is not None:
                    plt.savefig(f"{outpath}/rdf_{cosolvent_name}_{cosolvent_atom}.png")
                plt.close()
        
        # Finally do waters
        print("Analysing water")
        r_max = 8.5
        fig, ax = plt.subplots()
        plt.setp(ax, xlim=(0, r_max+1))
        irdf = rdf.InterRDF(oxygen_atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
        irdf.run(start=0, step=50)
        # irdf.run()
        ax.plot(irdf.results.bins, irdf.results.rdf, label="RDF")
        ax.set_xlabel(r'$r$ $\AA$')
        ax.set_ylabel("$g(r)$")
        ax.set_title(f"RDF-HOH O every 50 frames")
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        if outpath is not None:
            plt.savefig(f"{outpath}/rdf_HOH_O.png")
        plt.close()
        return

    def _autocorrelation(self, data):        
        """Gets the autocorrelation values.

        :param data: list of data for which the autocorrelation has to be computed.
        :type data: list
        :return: list of autocorrelations.
        :rtype: list
        """
        n = len(data)
        mean = np.mean(data)
        autocorr = correlate(data - mean, data - mean, mode='full', method='auto')
        return autocorr[n - 1:]
    
    def _plot_autocorrelation(self, data, ax, cosolvent_name1=None, cosolvent_atom1=None, cosolvent_name2=None, cosolvent_atom2=None):
        """Plots autocorrelations.

        :param data: list of data points for which we want to plot autocorrelations.
        :type data: list
        :param ax: matplotlib axis to add the autocorrelation to the RDF plot.
        :type ax: matplotlib.pyplot.axisß
        :param cosolvent_name1: name of the first cosolvent molecule, defaults to None
        :type cosolvent_name1: str, optional
        :param cosolvent_atom1: name of the first atom, defaults to None
        :type cosolvent_atom1: str, optional
        :param cosolvent_name2: name of the second cosolvent molecule, defaults to None
        :type cosolvent_name2: str, optional
        :param cosolvent_atom2: name of the second atom, defaults to None
        :type cosolvent_atom2: str, optional
        :return: the axis with the autocorrelation plot
        :rtype: matplotlib.pyplot.axis
        """
        title = f"{cosolvent_name1} {cosolvent_atom1}-{cosolvent_name2} {cosolvent_atom2}"
        data = data[0::2]
        autocorr_values = self._autocorrelation(data)
        # Normalize autocorrelation values for better plotting
        normalized_autocorr = autocorr_values / np.max(np.abs(autocorr_values))
        lags = np.arange(0, len(autocorr_values))
        pd.plotting.autocorrelation_plot(pd.Series(normalized_autocorr), ax=ax, label="Autocorrelation")
        ax.grid(False)
        ax.set_xlim([0, len(autocorr_values)])
        ax.set_title(title)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        return ax