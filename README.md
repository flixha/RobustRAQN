# RobustRAQN
**Robust** **R**equest **A**nd data **Q**uality **N**egotiater. This package detects seismic events through template matching using [EQcorrscan](https://github.com/eqcorrscan/EQcorrscan), but at the same time takes care of a wide range of challenges due to data quality issues, metadata errors, noise, legacy data, and specific seismic network setups. RobustRAQN can handle data from a distributed seismic network and one or more seismic arrays to achieve the most sensitive detections of seismic events.

**RobustRaqn is still in its Alpha state**, so the documentation, tests, tutorials, and API are still under development. You're welcome to try it out and don't hesitate to let me know if you need help with setting up a detection problem.

**Here is a list of some of the features:**
- performs checks of data quality based on IRIS Mustang data quality metrics (see [ispaq](https://github.com/iris-edu/ispaq) if you'd like to compute quality metrics for your own datasets)
- takes care of changes in the station setups (e.g., changes in instrument response) and network/station/location/channel codes
- takes care of mismatches, e.g. for the names of network, channel and location codes between seismic data and metadata in the most sensible way
- corrects the sampling rate of traces that differ slightly from nominal sampling rate
- reads in data in parallel (for now, focus on mseed data from a Seiscomp Data Structure)
- sets sensitive, adjustable thresholds to let users obtain even the smallest detections
- sets tested, adjustable limits for detections and picked events that sort out a lot of common misdetections in template matching (e.g., minimum number of stations, sites, P- / S-picks)
- includes all traces from a seismic array for which a pick or beam information is available into the templates, the matched-filter detection, and the lag-calc picking
- optionally sets station-specific filters derived from noise data
- optionally applies automatic gain control to reduce the prominence of spurious detections
- optionally weights channels according to RMS signal-to-noise ratio of template, and noise level during recording of template against noise level in data that is being searched 

**In addition, there is extra support for:**
- read and write input/output files for the location programs Growclust and Bayesloc
- sample scripts to deploy a detection run on a SLURM cluster with a two-level parallelization across multiple nodes.
- tested on a range of datasets to identify common challenges in template matching and handle many edge cases


**Here is what's still missing (but planned):**
- include automatic unit testing
- read IRIS Mustang metrics directly from SQL database or from webservice
- support alternative cross correlation metrics like C|C|
- support running from configuration file
- include detection benchmark sets


# INSTALL:

In a conda environment:
- `git clone https://github.com/flixha/RobustRAQN`
- `cd RobustRAQN`
- `conda install --file requirements.txt`
- (you may have to install wcmatch through pip)
- `python setup.py develop`

Some functionality may require EQcorrscan's `master ` branch, and some considerable speedups (e.g., for response correction) are not yet merged into obspy's master branch.
