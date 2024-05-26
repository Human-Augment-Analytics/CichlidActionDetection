import subprocess
import sys

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import hmmlearn, install if not available
try:
    import hmmlearn
except ImportError:
    install("hmmlearn")

import argparse
import subprocess
import datetime
import os

# Setup argument parser
parser = argparse.ArgumentParser(description='This script identifies regions of a large video that contain sand manipulation events')

# Input data arguments
parser.add_argument('--Movie_file', type=str, required=True, help='Name of movie file to analyze. Must be .mp4 video')
parser.add_argument('--Num_workers', type=int, default=0, help='Number of threads to run')
parser.add_argument('--Log', type=str, required=True, help='Log file to keep track of versions and parameters used')

# Temporary directories that will be deleted at the end of the analysis
parser.add_argument('--HMM_temp_directory', type=str, required=True, help='Location for temporary files to be stored. Avoid auto-synced locations like Dropbox')

# Output data arguments
parser.add_argument('--HMM_filename', type=str, required=True, help='Basename of output HMM files. ".txt" and ".npy" will be added to basename')
parser.add_argument('--HMM_transition_filename', type=str, required=True, help='Name of .npy file containing all transitions with associated magnitude')
parser.add_argument('--Cl_labeled_transition_filename', type=str, required=True, help='Name of .npy file containing all transitions assigned to clusters.')
parser.add_argument('--Cl_labeled_cluster_filename', type=str, required=True, help='Name of .csv file containing summary information for all clusters')
parser.add_argument('--Cl_videos_directory', type=str, required=True, help='Directory to hold video clips for all clusters')
parser.add_argument('--ML_frames_directory', type=str, required=True, help='Directory to hold frames to annotate for machine learning purposes')
parser.add_argument('--ML_videos_directory', type=str, required=True, help='Directory to hold videos to annotate for machine learning purposes')

# Parameters to filter when HMM is run
parser.add_argument('--VideoID', type=str, required=True, help='Short ID for the video')
parser.add_argument('--Video_framerate', type=float, required=True, help='Framerate of the video')
parser.add_argument('--Video_start_time', type=datetime.datetime.fromisoformat, required=True, help='Start time of the video')
parser.add_argument('--Filter_start_time', type=datetime.datetime.fromisoformat, help='Start time when the Clusters should be run')
parser.add_argument('--Filter_end_time', type=datetime.datetime.fromisoformat, help='End time when the Clusters should be run')

# Parameters for calculating HMM 
parser.add_argument('--HMM_blocksize', type=int, default=5, help='Blocksize (in minutes) to decompress video for HMM analysis')
parser.add_argument('--HMM_mean_window', type=int, default=120, help='Number of seconds to calculate mean over for filtering out large pixel changes for HMM analysis')
parser.add_argument('--HMM_mean_filter', type=float, default=7.5, help='Grayscale change in pixel value for filtering out large pixel changes for HMM analysis')
parser.add_argument('--HMM_window', type=int, default=10, help='Used to reduce the number of states for HMM analysis')
parser.add_argument('--HMM_seconds_to_change', type=float, default=1800, help='Used to determine probability of state transition in HMM analysis')
parser.add_argument('--HMM_non_transition_bins', type=float, default=2, help='Used to prevent small state transitions in HMM analysis')
parser.add_argument('--HMM_std', type=float, default=100, help='Standard deviation of pixel data in HMM analysis')

# Parameters for DBSCAN clustering
parser.add_argument('--Cl_min_magnitude', type=int, default=0, help='Transition magnitude to be included in cluster analysis')
parser.add_argument('--Cl_tree_radius', type=int, default=22, help='Tree radius for cluster analysis')
parser.add_argument('--Cl_leaf_num', type=int, default=190, help='Leaf number for cluster analysis')
parser.add_argument('--Cl_timescale', type=int, default=10, help='Tree radius for cluster analysis')
parser.add_argument('--Cl_eps', type=int, default=18, help='Eps for cluster analysis')
parser.add_argument('--Cl_min_points', type=int, default=90, help='Minimum number of points to create cluster')
parser.add_argument('--Cl_hours_in_batch', type=float, default=1.0, help='Number of hours to calculate cluster per batch')
#parser.add_argument('--Cl_neighbor_radius', type = int, default = 22, help = 'Tree radius for cluster analysis')

# Parameters for outputting video clips and frames for manual analysis and machine learning
parser.add_argument('--ML_frames_number', type=int, default=50, help='Number of frames to create for annotation for machine learning purposes')
parser.add_argument('--ML_videos_number', type=int, default=50, help='Number of videos to create for annotation for machine learning purposes')
parser.add_argument('--ML_videos_delta_xy', type=int, default=60, help='Half x and y size of each ML video created (in pixels)')
parser.add_argument('--ML_videos_manuallabel_delta_xy', type=int, default=100, help='Half x and y size of each manually labeled ML video created (in pixels)')
parser.add_argument('--ML_videos_delta_t', type=float, default=2, help='Half of t size of each ML video created (in seconds)')
parser.add_argument('--ML_videos_small_limit', type=int, default=500, help='Limit to prevent too many small videos being used for manual labeling')

args = parser.parse_args()

# Function to validate the input arguments
def check_args(args):
    bad_data = False
    if '.mp4' not in args.Movie_file:
        print('Movie_file must be an .mp4 file')
        bad_data = True
    if '.npy' not in args.HMM_transition_filename:
        print('HMM_transition_filename must have .npy extension')
        bad_data = True
    if '.npy' not in args.Cl_labeled_transition_filename:
        print('Cl_labeled_transition_filename must have .npy extension')
        bad_data = True
    if '.csv' not in args.Cl_labeled_cluster_filename:
        print('Cl_labeled_cluster_filename must have .csv extension')
        bad_data = True
    
    if bad_data:
        raise Exception('Error in argument input.')
    else:
        # Ensure temporary directories have a trailing slash and create them if they don't exist
        if args.HMM_temp_directory[-1] != '/':
            args.HMM_temp_directory += '/'
        if os.path.exists(args.HMM_temp_directory):
            subprocess.run(['rm', '-rf', args.HMM_temp_directory])
        os.makedirs(args.HMM_temp_directory)
        
        if args.Cl_videos_directory[-1] != '/':
            args.Cl_videos_directory += '/'
        if not os.path.exists(args.Cl_videos_directory):
            os.makedirs(args.Cl_videos_directory)
        
        if args.ML_frames_directory[-1] != '/':
            args.ML_frames_directory += '/'
        if not os.path.exists(args.ML_frames_directory):
            os.makedirs(args.ML_frames_directory)
        
        if args.ML_videos_directory[-1] != '/':
            args.ML_videos_directory += '/'
        if not os.path.exists(args.ML_videos_directory):
            os.makedirs(args.ML_videos_directory)
        
        for ofile in [args.HMM_filename, args.HMM_transition_filename, args.Cl_labeled_transition_filename, args.Cl_labeled_cluster_filename, args.Log]:
            odir = os.path.dirname(ofile)
            if not os.path.exists(odir) and odir != '':
                os.makedirs(odir)

check_args(args)

# Write the log file with the arguments and system information
with open(args.Log, 'w') as f:
    for key, value in vars(args).items():
        print(f"{key}: {value}", file=f)
    print(f'PythonVersion: {sys.version.replace("\n", " ")}', file=f)
    import pandas as pd
    print(f'PandasVersion: {pd.__version__}', file=f)
    import numpy as np
    print(f'NumpyVersion: {np.__version__}', file=f)
    import hmmlearn
    print(f'HMMLearnVersion: {hmmlearn.__version__}', file=f)
    import scipy
    print(f'ScipyVersion: {scipy.__version__}', file=f)
    import cv2
    print(f'OpenCVVersion: {cv2.__version__}', file=f)
    import sklearn
    print(f'SkLearnVersion: {sklearn.__version__}', file=f)
    print(f'Username: {os.getenv("USER")}', file=f)
    print(f'Nodename: {os.uname().nodename}', file=f)
    print(f'DateAnalyzed: {datetime.datetime.now()}', file=f)


# Filter out HMM related arguments
HMM_args = {key: value for key, value in vars(args).items() if 'HMM' in key or 'Video' in key or 'Movie' in key or 'Num' in key or 'Filter' in key}

# Construct and run HMM command
HMM_command = ['python3', 'Utils/calculateHMM.py']
for key, value in HMM_args.items():
    HMM_command.extend([f'--{key}', str(value)])

HMM_command = ['python3', 'Utils/calculateHMM.py']
for key, value in HMM_args.items():
	HMM_command.extend(['--' + key, str(value)])

print(HMM_command, flush=True)
subprocess.run(HMM_command)

# Filter out clustering related arguments
cluster_args = {key: value for key, value in vars(args).items() if 'HMM' not in key or 'filename' in key}
cluster_args.pop('Log', None)

# Construct and run clustering command
cluster_command = ['python3', 'Utils/calculateClusters.py']
for key, value in cluster_args.items():
    cluster_command.extend([f'--{key}', str(value)])

subprocess.run(cluster_command)



