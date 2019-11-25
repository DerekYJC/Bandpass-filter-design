# Bandpass-filter-design
Evaluate the bandpass filter with Butterworth filter. Here I used [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html).

Different frequency bands may require **different orders** to design the filter. The frequency response figure is shown.

In this section, I evaluate two different filter designs.
- *Version 1* : **Lowpass filter + Highpass filter**
- *Version 2* : **Bandpass filter**

To optimize the filter performance for narrower bandwidth, there I used [FIR filter](https://scipy-cookbook.readthedocs.io/items/FIRFilter.html).

- *Version 3* : **FIR Bandpass filter**
