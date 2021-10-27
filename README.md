# Threerra: A Qiskit module for three-level systems

### Repository description

This repository holds the in-development source code implementation for [the threerra project](https://qiskitfallfest.hypeinnovation.com/servlet/hype/IMT?documentTableId=396317851978759375&userAction=Browse&templateName=&documentId=f475bb33e0e0758a98b1c90d754aeab4), in the context of the first [Quantum Hackathon CIC- IPN 2021 (Mexico)](https://qiskitfallfest.hypeinnovation.com/servlet/hype/IMT?documentTableId=396317851978733212&userAction=Browse&templateName=&documentId=184ef5cd6b1e8c527512c0231f5f474a).

### Motivation and intention

The Qiskit Terra module provides the necessary tools to compose quantum programs at the level of quantum circuits and pulses. While pulses allow to access hardware as higher-level systems, the circuit composer is currently limited to two-levels systems (qubits).

Our goal is to create a module called ***threerra***, to allow users to both create unitary operations acting onto three-level systems (qutrits) using Qiskit Pulse and to execute them on real hardware available through the IBM Quantum platform.

## Example notebooks

### Can I see some examples of this package in action?

Some jupyter notebooks with examples using `threerra` are contained in the folder `Example_notebooks/`. Remember to have `threerra` installed on your system before running the notebooks!

### Can I download the example notebooks to experiment locally?

Yes! The folder `Example_notebooks/` can be downloaded as a compressed zip file from [this link](https://gitlab.com/jgidi/threerra/-/archive/master/threerra-master.zip?path=Example_notebooks).


## How to get this package?

### Installation

This package can be installed via pip by running

```sh
pip install git+https://github.com/jgidi/threerra
```
    
or, alternatively, by running

```sh
python -m pip install git+https://github.com/jgidi/threerra
```
    
### Uninstallation

In the same manner, the `threerra` package can be uninstalled as

```sh
pip uninstall threerra
```

or, equivalently,

```sh
python -m pip uninstall threerra
```

# License notice

As expressed on the `LICENCE` file, we make use of the permissive Apache License 2.0.

Also, this project makes use of several well known python libraries; [qiskit](https://qiskit.org/), [numpy](https://numpy.org/), [sklearn](https://qiskit.org/), [scipy](https://www.scipy.org/) and [matplotlib](https://matplotlib.org/). We acknowledge their developers and express full compliment to their respective licenses.
