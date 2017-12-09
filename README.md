# environment setup for windows  
```
conda create -n pytorch python=3.6 scipy numpy matplotlib nltk
# this is a custom build pytorch. version 0.2.1
conda install -n pytorch -c peterjc123 pytorch
```
you may need to modify source code at torch/backends/cudnn/__ init__.py  
add this code to line number 20  
```
__cudnn_version = lib.cudnnGetVersion()
```  

