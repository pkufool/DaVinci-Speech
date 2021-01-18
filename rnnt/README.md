#### install warpRNNT (for RNNTLoss)
* build warp transducer
```
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake ..
make
```
* set env
```
export CUDA_HOME=$HOME/tools/cuda-9.0 # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
```
* install pytorch binding
```
cd pytorch_binding
python setup.py install
```
