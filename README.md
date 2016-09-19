
The install process assumes that CUDA and CUDNN are installed.

The build_env script assumes you are using Ubuntu.  If you are not, you
are on your own.

The script build_env builds the environment for running the demo
This should be turned into a container someday.

- Creating the environment

# virtualenv venv
# source venv/bin/activate
(venv) # sh build_env

Running the Demo:

(venv) # ./demo_video_infer.py <pedestrian video>

Leaving the environment

(venv) # deactivate
#






