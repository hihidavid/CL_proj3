## environment setup for windows  
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


## discuss  
1. possible improve:  
    1. bi-directional GRU
    2. peephole(LSTM only): allow gates depends not only on previous hidden state but also on previous memory/internal station
    3. stack of GRU: multiple layers. promote non-linearity. generally improve accuracy. more layers are unlikely to make a big difference and may lead to overfitting.
    4. Truncated BPTT: only backprop constant steps. Help with runtime.
    5. batch update:
    
    * links that have some suggestion:  
    [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)  

## resources
* basics:  
[lecture slide seq2seq](http://www.cs.umd.edu/class/fall2017/cmsc723/slides/slides_16.pdf)  
[lecture slide attention](http://www.cs.umd.edu/class/fall2017/cmsc723/slides/slides_17.pdf)  
[lecture slide LSTM](http://mt-class.org/jhu/slides/lecture-nn-lm.pdf)  
[understanding LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[Implementing a GRU/LSTM](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)  

* improve:






