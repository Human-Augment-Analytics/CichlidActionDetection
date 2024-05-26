# CichlidActionDetection

This repository contains code for analyzing videos of Lake Malawi male cichlids building sand bowers to attract female mates in naturalistic environments. Using Hidden Markov Models (HMMs) to identify long-lasting changes in pixel color, followed by spatial, distance-based clustering using DBSCAN, individual sand manipulation events are identified, including scoops and spits using the fish's mouth, along with other sand changes caused by fins and the body.

## Files and Scripts

#### `testScript.py`
An example script demonstrating how the analysis can be run.

#### `VideoFocus.py`
The master script for analyzing an MP4 movie file (typically 10 hours long with a frame rate of 30 frames/second). It identifies parts of the video where fish manipulate sand using their mouth, body, or fins, resulting in the creation of short video clips containing each sand manipulation event. This script performs the following steps:
1. **HMM Analysis**: Analyzes pixel color through time to identify time points when fish manipulate sand in a single pixel.
2. **Cluster Analysis**: Groups nearby pixels together that have been manipulated in the same event.
3. **Clip Creation**: Generates videos surrounding each cluster.

The created videos can be used for 3D ResNet-based classification into 10 different types of sand manipulation actions. The repository for this classification code is contained elsewhere.

Arguments for this script include the ability to parallelize its execution using the `Num_workers` argument, specify output file names and locations, and modify key parameters for each aspect of the analysis. This script also creates video and frame files that can be used for manual annotation.

#### `VideoFocus.yaml`
Anaconda environment file for running this repository.

#### `Utils/calculateHMM.py`
Master script for calculating HMM values for each pixel. This script:
1. Decompresses an MP4 video file into numpy arrays for each row of data, containing pixel values at 1 frame/sec. This data is stored in a large temporary directory that is deleted at the end of the analysis.
2. Runs HMM analysis on each numpy array.
3. Converts each HMM transition to a coordinate file used for cluster analysis.

#### `Utils/calculateClusters.py`
Master script for calculating clusters from HMM transitions. This script:
1. Clusters HMM transitions into groups, assigning a clusterID for each transition. A `-1` cluster ID indicates transitions that could not be clustered.
2. Generates video clips surrounding each cluster.
3. Generates pictures of random frames taken from the video that can be used for manual annotation.

#### `Utils/Decompress_block.py`
Utility script used by `calculateHMM.py` to decompress the MP4 file into individual numpy arrays for each row of the video.

#### `Utils/HMM_Analyzer.py`
Utility class used for storing and analyzing HMM data.

#### `Utils/HMM_row.py`
Utility script used by `calculateHMM.py` to calculate HMMs for each numpy row file.

#### `Utils/createClip.py`
Utility script used by `calculateClusters.py` to create clusters in specified time blocks.