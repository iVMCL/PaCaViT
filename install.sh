#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: install virtual_env_name"
    exit
fi

if [[ $OSTYPE != 'linux-gnu'* ]]; then
    echo "LINUX GNU OS needed (e.g., Ubuntu 20.04)."
    exit  
fi

VENV_NAME=$1
UPDATE_PILLOW=1

# path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "---------------------- Work in $DIR ---------------------- "

# check if the env exists
find_conda_env(){
    conda env list | grep -w ${VENV_NAME} >/dev/null 2>/dev/null
}
# remove: conda env remove -n env_name

# update conda 
# conda update -n base -c defaults conda

# check env before install
if ! find_conda_env; then   
    echo "---------------------- Createing the conda env ${VENV_NAME}..."
    conda env create -n ${VENV_NAME} -f "$DIR"/environment.yaml      
# else
#     echo "Updating the conda env ${VENV_NAME}..."
#     conda env update -n ${VENV_NAME} -f "$DIR"/environment.yaml      
fi


function find_conda_package {
    conda list | grep -w "$1" >/dev/null 2>/dev/null
}

# mmengine, mmcv, mmdet, mmseg, mmpretrain
if [ $CONDA_DEFAULT_ENV != ${VENV_NAME} ] ; then
    echo "---------------------- Not inside the virtual env $VENV_NAME ---------------------- "
    echo "---------------------- Please manually run: conda activate $VENV_NAME ------------- "
    echo "----------------------  and then re-run this installation script ------------------ "
    exit
else
    if find_conda_package mmengine; then
        echo "mm packages installed already"
    else
        pip install -U openmim
        mim install mmengine        
        mim install "mmcv>=2.0.0rc4"
        mim install "mmdet>=3.0.0rc0"
        mim install "mmsegmentation>=1.0.0"
        mim install "mmpretrain>=1.0.0rc7"
        python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
    fi
fi

# update pillow, https://fastai1.fast.ai/performance.html#faster-image-processing
if [ $UPDATE_PILLOW == 1 ]; then 
    if find_conda_package pillow-simd; then
        echo "PILLOW-SIMD installed already"
    else
        # check env before install
        if [ $CONDA_DEFAULT_ENV != ${VENV_NAME} ] ; then
            echo "---------------------- Not inside the virtual env $VENV_NAME ---------------------- "
            echo "---------------------- Please manually run: conda activate $VENV_NAME ------------- "
            echo "----------------------  and then re-run this installation script ------------------ "
            exit
        fi

        echo "---------------------- Install Pillow-SIMD for Faster Image Processing ---------------------- "
        echo "    If errors occur, please contact your admin to install prerequistes for pillow https://pillow.readthedocs.io/en/stable/installation.html#building-on-linux"
        ## prerequistes for pillow https://pillow.readthedocs.io/en/stable/installation.html#building-on-linux
        ##   which are needed to be installed if some errors occur in installing the pillow-simd
        ##   check with the system admin for the installation
        
        # sudo apt-get install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev libfreetype6-dev \
        #   liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev libfribidi-dev \
        #   libxcb1-dev
        # sudo apt-get install gcc-multilib

        conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
        pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
        conda install -yc conda-forge libjpeg-turbo
        CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
        conda install -y -c zegami libtiff-libjpeg-turbo
        conda install -y jpeg libtiff        
    fi
fi



