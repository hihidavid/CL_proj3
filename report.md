[markdown cheetsheet](https://guides.github.com/features/mastering-markdown/)  
# Part I  
## Q1  
* Describe the network architecture for the encoder:  


* for the decoder model:  
* what kind of RNNs are used?  

* What are the dimensions of the various layers?  

* What are the non-linearity functions used?  

* How is the attention computed? 


## Q2: describe hyperparameters  
* n_iters:  


* learning_rate: 


## Q3 select hyperparameters  
* Describe the experiments you ran to select those values, and explain your reasoning.  


## Q4 and Q5 teacher forcing  
* Explain how training works if teacher_forcing is set to 0, and if teacher_forcing is set to 1  


* Investigate the impact of teacher forcing empirically. Report learning curves for 0.1, 0.5 and 0.9, and explain what you observe.


## Q6 attention model  
* why the attention model is useful to model transliteration  


## Q7 noattention.py  
* use a sequence-to-sequence model without attention  


## Q8 comparision  
comparing the behavior of the sequence-to-sequence model with and without attention empirically, at training and test time  


# Part II  
## Q9 explain what you did  
* Define the problem that you are addressing  

* Explain why this problem matters  

* Describe your proposed solution  

* Explain how your solution addresses the problem  

   
## Q10   

Design an experiment to test whether your solution successfully addresses the problem (e.g., compare the performance of your new model with the baseline system on the validation set, as well as the learning curves.). Present and discuss your results.  If the results are unexpected, explain what you think went wrong, and provide supporting analysis. 


## Q11  
Provide the implementation so we can replicate your results. You can create additional files as needed as long as they are in the p3 directory. Instructions for running the code should be provided in the readme file. Note that it is your responsibility to test your code and make sure the instructions are accurate and self-contained.  If the code doesn't run, we will not attempt to debug it.  By default, we will evaluate by running the following command to train a transliteration model, and evaluate it on held out test data:


# Part III  
## extra credit  
will be used to reward groups that experiment with ambitious ideas that require substantially more work and deeper understanding of the model (e.g., successfully using reinforcement learning or minimum risk training to incorporate edit distance during training would lead to extra-credit.)


