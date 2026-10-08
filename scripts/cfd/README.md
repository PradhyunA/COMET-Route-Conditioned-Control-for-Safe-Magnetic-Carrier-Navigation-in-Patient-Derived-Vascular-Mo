# SimVascular CFD — 0082_H_PULM_H

These scripts reproduce the CFD run used for the pulmonary anatomy.

## Case

Use the SimVascular simulation directory:

    0082_H_PULM_H/Simulations/0084_0001

The anatomy ZIP already contains the mesh, boundary conditions, inflow waveform,
initial restart, and solver configuration.

## CFD settings

The supplied/native pulmonary settings are retained:

- Time step: `0.0002375 s`
- Total timesteps: `8125`
- Cardiac period: approximately `0.9524 s`
- Approximately 4010 timesteps per cardiac cycle
- Approximately two cardiac cycles total
- Restart output every `401` timesteps

The second cardiac cycle is sampled at:

    4010
    4411
    4812
    5213
    5614
    6015
    6416
    6817
    7218
    7619
    8020

This produces 11 approximately uniformly phase-spaced CFD frames.

## Run CFD

The scripts assume `mpiexec`, `svsolver`, and `python3` are available.

Example with 6 MPI ranks:

    ./scripts/cfd/run_0082_cfd.sh \
      /path/to/0082_H_PULM_H/Simulations/0084_0001 \
      6

Example with 24 MPI ranks:

    ./scripts/cfd/run_0082_cfd.sh \
      /path/to/0082_H_PULM_H/Simulations/0084_0001 \
      24

If the executables are not on PATH:

    MPIEXEC=/path/to/mpiexec \
    SVSOLVER=/path/to/svsolver \
    ./scripts/cfd/run_0082_cfd.sh \
      /path/to/0082_H_PULM_H/Simulations/0084_0001 \
      6

## Postprocess cycle 2

After the CFD run finishes:

    ./scripts/cfd/postprocess_0082_cycle2.sh \
      /path/to/0082_H_PULM_H/Simulations/0084_0001

If `svpost` is not on PATH:

    SVPOST=/path/to/svpost \
    ./scripts/cfd/postprocess_0082_cycle2.sh \
      /path/to/0082_H_PULM_H/Simulations/0084_0001

The resulting VTUs are written to:

    post_cycle2_4010_8020/

Do not commit restart files, VTUs, solver logs, or full CFD output directories to Git.
