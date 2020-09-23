Signal pipes (aka *sigpipes*) are Python library for processing of digital 
representation of signals.

The library is focused on processing physiological signal (e.g. EKG or EMG) but
is basic design supports processing of any multichannel signals.

 The library is based on two mechanism:
 
 1. flexible **containers** for signals and their metadata in the form of hierarchical
 dictionary with a set of "well known" keys forming solid skeleton of data 
 representation.
 
 2. **pipelines** that are formed by operators for processing, storing and visualization 
 og signals. The pipelines are not limited to simple linear chains. The more complex
 forms are supported (branching, alternatives, parallel processing).
 
 Current version support these tools in form of pipeline operators.
 
*  readers of physionet.org databases
*  readers of matlab files from `Megawin` application
*  exporter to `pandas` dataframes
*  configurable plotting using `matplotlib`
*  selection of channels or subsamples
*  extraction of signal features
*  signal adjustment (any unary ufunc, convolution, cross correlation, polyphase resampling) 
   based on scipy.signals and numpy
*  serialization to HDF5 