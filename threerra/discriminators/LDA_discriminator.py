#!/usr/bin/env python3

import os
import numpy as np
from threerra import pulses
import qiskit.pulse as pulse
from qiskit.tools.monitor import job_monitor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

lda_discriminator_data = np.zeros((1, 2))
datapath = os.path.join(os.path.dirname(__file__),
                        'lda_discriminator_data.npy')

def load_datafile():
    global lda_discriminator_data
    try:
        lda_discriminator_data = np.load(datapath)
    except FileNotFoundError:
        print("Data file not found.\
        \nRegenerate it with 'threerra.discriminators.LDA_discriminator.train_discriminator(qc : QuantumCircuit3)'")

def train_discriminator(qc3, shots=1024):

    # Pulses
    pi_pulse_01 = pulses.pi_pulse_01_sched(qc3)
    pi_pulse_12 = pulses.pi_pulse_12(qc3)
    measure_pulse = pulses.measure(qc3)

    # Schedule to generate G state
    schedule0 = pulse.Schedule()
    schedule0 |= measure_pulse

    # Schedule to generate E state
    schedule1 = pulse.Schedule()
    schedule1 |= pi_pulse_01
    schedule1 |= measure_pulse << schedule1.duration

    # Schedule to generate F state
    schedule2 = pulse.Schedule()
    schedule2 |= pi_pulse_01
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
    results_data = np.concatenate([results.get_memory(i)[:, qc3.qubit]
                                   for i in range(len(results.results))])

    # Reorder as real pairs
    results_data_pairs = np.array([[np.real(dat), np.imag(dat)]
                             for dat in results_data])

    global lda_discriminator_data
    lda_discriminator_data = results_data_pairs

    np.save(datapath, lda_discriminator_data)

    load_datafile()

    # Estimate accuracy
    references = np.concatenate((np.zeros(shots), np.ones(shots), 2*np.ones(shots)))
    values = discriminator(results_data)

    accuracy = np.mean(references == values)
    print(f'Discriminator achieved an accuracy of {round(100*accuracy, 2)}% after training.')


def discriminator(data, shots=1024):

    points = np.array([[np.real(dat), np.imag(dat)] for dat in data])

    # construct vector w/ 0's, 1's and 2's (for testing)
    states = np.concatenate((np.zeros(shots), np.ones(shots), 2*np.ones(shots)))

    # Shuffle and split data into training and test sets
    IQ_train, _, states_train, _ = train_test_split(lda_discriminator_data, states, test_size=0.5)

    # Set up the LDA
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(IQ_train, states_train)

    counts = np.round(LDA.predict(points)).astype(int)

    return counts

load_datafile()
