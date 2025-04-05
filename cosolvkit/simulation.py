import os
from sys import stdout
import openmm
import openmm.app as app
import openmm.unit as openmmunit

def run_simulation( results_path: str = "output",
                    pdb_fname: str = None,
                    system_fname: str = None,
                    membrane_protein: bool = False, 
                    traj_write_freq: int = 25000,
                    time_step: float = 0.004,
                    temperature: float = 300,
                    simulation_steps: int = 25000000, # 100ns at 4fs time step
                    seed: int = None
                    ):
    """_summary_

    :param results_path: path to where to save the results, defaults to "output"
    :type results_path: str, optional
    :param pdb_fname: path to the pdb file of the system, defaults to None
    :type pdb_fname: str
    :param system_fname: path to the system.xml generated by Cosolvkit is simulation format was OpenMM, defaults to None.
    :type system_fname: str
    :param membrane_protein: True if using a membrane in the system, False otherwise
    :type membrane_protein: bool, optional
    :param traj_write_freq: frequency of writing the trajectory, defaults to 25000
    :type traj_write_freq: int, optional
    :param time_step: time step of the simulation, defaults to 0.004
    :type time_step: float, optional
    :temperature: temperature of the simulation, defaults to 300 K.
    :type temperature: float, optional
    :param simulation_steps: number of simulation steps, defaults to 25000000
    :type simulation_steps: int, optional
    :param seed: random seed for reproducibility, defaults to None
    :type seed: int, optional
    :raises ValueError: different checks are performed and expections are raised if some of the fail.
    """

    # set up parameters and units
    pressure = 1 * openmmunit.bar # bar
    Tstart = 50 * openmmunit.kelvin
    Tend = temperature * openmmunit.kelvin
    Tstep = 5 * openmmunit.kelvin
    warming_steps = 100000
    warming_timestep = 0.001 * openmmunit.picoseconds # 1fs
    warming_time = warming_steps * warming_timestep #ps
    time_step = time_step * openmmunit.picoseconds # 4fs
    production_time = simulation_steps * time_step #ps
    total_steps = warming_steps + simulation_steps
    
    assert pdb_fname is not None and system_fname is not None, "To run a simulation in OpenMM both a pdb file and system.xml must be provided."
    pdb = app.PDBFile(pdb_fname)
    topology = pdb.topology
    positions = pdb.positions
    system = openmm.XmlSerializer.deserialize(open(system_fname).read())

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = openmm.Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")

    integrator = openmm.LangevinMiddleIntegrator(Tstart,
                                                1 / openmmunit.picosecond,
                                                warming_timestep)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    
    simulation = app.Simulation(topology, system, integrator, platform)
        
    print('Adding reporters to the simulation')
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(os.path.join(results_path, "statistics.csv"), traj_write_freq, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps))
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(stdout, traj_write_freq, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
    
    #every 0.1ns
    simulation.reporters.append(app.DCDReporter(os.path.join(results_path, "trajectory.dcd"),
                                            reportInterval=traj_write_freq, enforcePeriodicBox=True))

    
    #every 1ns
    simulation.reporters.append(app.CheckpointReporter(os.path.join(results_path,"simulation.chk"), traj_write_freq*10)) 

    print("Setting initial positions")
    simulation.context.setPositions(positions)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    print(f'Heating system in NVT ensemble for {warming_time} ps')
    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)
    
    # Warm up the system gradually, i.e., temperature annealing
    for i in range(nT):
        temperature = Tstart.value_in_unit(openmmunit.kelvin) + i * Tstep.value_in_unit(openmmunit.kelvin)
        integrator.setTemperature(temperature)
        print(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    # Increase the timestep for production simulations
    integrator.setStepSize(time_step)

    print(f'Adding a Montecarlo Barostat to the system')
    if membrane_protein:
        barostat = openmm.MonteCarloMembraneBarostat(pressure,  # Pressure in bar 
                                                     0.0 * openmmunit.nanometers * openmmunit.bar,  # surface tension 
                                                     Tend,  # Temperature in Kelvin 
                                                     openmm.MonteCarloMembraneBarostat.XYIsotropic,  # XY isotropic scaling 
                                                     openmm.MonteCarloMembraneBarostat.ZFree,  # Z dimension is free 
                                                     15  # Number of Monte Carlo steps 
                                                     ) 
        system.addForce(barostat)
    else:
        system.addForce(openmm.MonteCarloBarostat(pressure, Tend))
    simulation.context.reinitialize(preserveState=True)

    print(f"Running simulation in NPT ensemble for {production_time/1000} ns")
    simulation.step(simulation_steps) 
    return