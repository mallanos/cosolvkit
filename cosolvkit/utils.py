#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Utils functions
#

import contextlib
import os
import tempfile
import shutil
import subprocess
import importlib
from collections import defaultdict

import numpy as np
import mdtraj
from MDAnalysis import Universe
from openmm.app import *
from openmm import *
import pdbfixer

class MutuallyExclusiveParametersError(Exception):
        pass   


def add_harmonic_restraints(prmtop, inpcrd, system, atom_selection, k=1.0):
    """Add harmonic restraints to the system.

    Args:
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object
        system (System): OpenMM system object created from the Amber topology object
        atom_selection (str): Atom selection (see MDTraj documentation)
        k (float): harmonic force restraints in kcal/mol/A**2 (default: 1 kcal/mol/A**2)

    Returns:
        (list, int): list of all the atom ids on which am harmonic force is applied, index with the System of the force that was added

    """
    mdtop = mdtraj.Topology.from_openmm(prmtop.topology)
    atom_idxs = mdtop.select(atom_selection)
    positions = inpcrd.positions

    # Tranform constant to the right unit
    k = k * unit.kilocalories_per_mole / unit.angstroms**2
    k = k.value_in_unit_system(unit.md_unit_system)

    if atom_idxs.size == 0:
        print("Warning: no atoms selected using: %s" % atom_selection)
        return ([], None)

    # Take into accoun the periodic condition
    # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomExternalForce.html
    force = CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    
    harmonic_force_idx = system.addForce(force)
    
    force.addGlobalParameter("k", k)
    print(force.getGlobalParameterDefaultValue(0))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for atom_idx in atom_idxs:
        #print(atom_idx, positions[atom_idx].value_in_unit_system(units.md_unit_system))
        force.addParticle(int(atom_idx), positions[atom_idx].value_in_unit_system(unit.md_unit_system))

    return atom_idxs


def update_harmonic_restraints(simulation, k=1.0):
    """Update harmonic restraints force
    
    Args:
        simulation (Simulation): OpenMM simulation object
        k (float): new harmonic force constraint in kcal/mol/A**2 (default: 1 kcal/mol/A**2)

    """
    # Tranform constant to the right unit
    k = k * unit.kilocalories_per_mole / unit.angstroms**2
    k = k.value_in_unit_system(unit.md_unit_system)
    
    simulation.context.setParameter('k', k)

def fix_pdb(pdbfile: str, pdbxfile: str, keep_heterogens: bool=False) -> tuple[Topology, list]:
    """ Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - remove nonstandard residues
    
    Args:
        pdbfile (str): pdb string old format
        pdbxfile (str): pdb string new format
        keep_heterogens (bool): if False all the heterogen atoms but waters are deleted.
                                Defaults to False.
    """
    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile, pdbxfile=pdbxfile)
    fixer.findMissingResidues()
    
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms() 
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    return fixer.topology, fixer.positions
    
def add_variants(topology: Topology, positions: list, variants: list=list()) -> tuple[Topology, list]:
    """Adds variants for specific protonation states.

    Args:
        topology (Topology): openmm topology
        positions (list): openmm positions
        variants (list): list of variants to apply for the protonation states. 
                         The ones to leave untouched are passed as None. Defaults to list().

    Returns:
        tuple[Topology, list]: topology and positions with added protonation states.
    """
    modeller = Modeller(topology, positions)
    added_variants = modeller.addHydrogens(variants=variants)
    return modeller.topology, modeller.positions
