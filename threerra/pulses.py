#!/usr/bin/env python3

import numpy as np

from qiskit import pulse
import qiskit.pulse.library as pulse_lib

def gen_pulse(qc3, shift_phase=0, angle=0, sideband_freq=0, name: Optional[str] = None):
    """
    Generates a pulse with given phase, angle and sideband frequency.
    The parameters must be in radians i.e (pi/s), with s in [0, 2pi].

        Args:
            qc3: QuantumCircuit3 circuit.
            shift_phase: The channel that will acquire data.
            angle: The classical memory slot in which to store the classified readout result.
            sideband_freq: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            name: Name of the pulse.

        Returns:
            Generated pulse with the desired parameters.
            
     """

    if name is None:
        name = ''

    # Init pulse
    schedule = pulse.Schedule(name=name)

    # Possible phase shift
    if not shift_phase:
        schedule |= pulse.ShiftPhase(shift_phase, qc3.drive_chan)

    # Rotation and/or frequency shift
    if (not angle) or (not sideband_freq):
        base_pulse = pulse_lib.gaussian(
            duration=qc3.drive_samples,
            sigma=qc3.drive_sigma,
            amp=qc3.pi_amp_01*angle/np.pi,
            name=name,
        )
        if not sideband_freq:
            base_pulse = qc3.sideband_pulse(base_pulse, sideband_freq,
                                            name=name)

        schedule |=  pulse.Play(base_pulse, qc3.drive_chan)

    return schedule


def measure(qc3):
    """
    Generates a measurement pulse on the given qubits, using the backend's default configuration.
    
        Args:
            qc3: QuantumCircuit3 circuit.

        Returns:
            Generated measurement pulse.
            
    """
    meas_idx = [qc3.qubit in group
                       for group in qc3.backend_config.meas_map].index(True)

    pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
    )

    return pulse

def pi_pulse_01(qc3):
    """
    Generates a 0 --> 1 pi pulse on the given qubits, using previously defined parameters for duration, pi amplitude, etc.
    
        Args:
            qc3: QuantumCircuit3 circuit.

        Returns:
            Generated 0 --> 1 pi pulse.
    """
    pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                               amp=qc3.pi_amp_01,
                               sigma=qc3.drive_sigma,
                               name='x_01')
    return pulse

def pi_pulse_12(qc3):
    """
    Generates a 1 --> 2 pi pulse on the given qubits with the sideband method,
    using the previously defined parameters for duration, pi amplitude, sideband frequency, etc.
    
        Args:
            qc3: QuantumCircuit3 circuit.

        Returns:
            Generated 0--> 2 pi pulse.
    """

    pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                               amp=qc3.pi_amp_12,
                               sigma=qc3.drive_sigma,
                               name='x_12')

    # Shift frequency by sidebanding
    pulse = qc3.apply_sideband(pulse, qc3.qubit_freq_est_12,
                               name="x_12")

    return pulse
