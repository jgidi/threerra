#!/usr/bin/env python3

import os
import numpy as np
from threerra import pulses
from qiskit import pulse
from qiskit.tools.monitor import job_monitor

nearest_discriminator_data = np.zeros(3)
datapath = os.path.join(os.path.dirname(__file__),
                        'nearest_discriminator_data.npy')

def load_datafile():
    global nearest_discriminator_data
    try:
        nearest_discriminator_data = np.load(datapath)
    except FileNotFoundError:
        print("Data file not found.\
        \nRegenerate it with 'threerra.discriminators.nearest_discriminator.train_discriminator(qc : QuantumCircuit3)'")


def train_discriminator(qc3, shots=1024):

    # Pulses
    pi_pulse_01 = pulses.pi_pulse_01(qc3)
    pi_pulse_12 = pulses.pi_pulse_12(qc3)
    measure_pulse = pulses.measure(qc3)

    # Schedule to generate G state
    schedule0 = pulse.Schedule()
    schedule0 |= measure_pulse

    # Schedule to generate E state
    schedule1 = pulse.Schedule()
    schedule1 |= pulse.Play(pi_pulse_01, qc3.drive_chan)
    schedule1 |= measure_pulse << schedule1.duration

    # Schedule to generate F state
    schedule2 = pulse.Schedule()
    schedule2 |= pulse.Play(pi_pulse_01, qc3.drive_chan)
    schedule2 |= pulse.Play(pi_pulse_12, qc3.drive_chan) << schedule2.duration
    schedule2 |= measure_pulse << schedule2.duration

    schedules = [schedule0, schedule1, schedule2]

    # Generate desired states
    job = qc3.backend.run(schedules,
                       meas_level=1,
                       meas_return='single',
                       shots=shots,
                       schedule_los=[{qc3.drive_chan: qc3.qubit_freq_est_01}] * 3)

    print('Training discriminator...')
    job_monitor(job)

    results = job.result(timeout=120)
    results_data = [results.get_memory(i)[:, qc3.qubit]
                    for i in range(len(results.results))]

    # find the centroid of each distribution of IQ values
    centroids = np.array([np.mean(IQvalue) for IQvalue in results_data])

    # Save centroids found to disk
    np.save(datapath, centroids)

    # Reload
    load_datafile()

    # Estimate accuracy
    references = np.concatenate((np.zeros(shots), np.ones(shots), 2*np.ones(shots)))
    values = discriminator(np.concatenate(results_data))

    accuracy = np.mean(references == values)
    print(f'Discriminator achieved an accuracy of {round(100*accuracy, 2)}% after training.')


def discriminator_single_point(point):

    dist = point - nearest_discriminator_data

    return np.abs(dist).argmin()


def discriminator(points):
    return [discriminator_single_point(point) for point in points]


load_datafile()
