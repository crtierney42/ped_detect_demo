
The install process assumes that CUDA and CUDNN are installed.

The build_env script assumes you are using Ubuntu.  If you are not, you
are on your own.

The script build_env builds the environment for running the demo
This should be turned into a container someday.  If you have the following
dependencies satisified, you do not have to run the build_env script

- nvcaffe (at least 0.15.x)
- opencv (at least 2.4.11 with ffmpeg support built in)
- the various python packages referenced in requirements.txt
- the various packages referenced in build_env

- Creating the environment

  # virtualenv venv
  # source venv/bin/activate
  (venv) # sh build_env

Running the Demo:

  (venv) # ./demo_video_infer.py <pedestrian video>

Leaving the environment

  (venv) # deactivate
  #






