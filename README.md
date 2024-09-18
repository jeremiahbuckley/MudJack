# MudJack  
  
This is a testing framework for ML libraries. Intended targets: PyTorch, TensorFlow and JAX.  
  
The inspiration for this testing benchmark is this paper by Mince, Dinh, Kgomo, Thompson and Hooker (The Grand Illusion: The Myth of Software Portability and Implications for ML Progress)[https://arxiv.org/abs/2309.07181]. Apparently ML libraries do not work across different chipset architectures.  
  
This code is intended to be run against all of these frameworks and report on errors and performance differences.  
  
The planned architecture:  
1. Basic tests of functionality in all 3 languages, 3 different runtimes.  
2. These will be spun up in separate Container runtimes with an api endpoint.  
3. On calling the api endpoint, they will return a file (probably json, maybe csv?) with the results of the test.  
4. Using a Kubernetes framework and correctly managed nodes with the appropriate chips, we should be able to run this test across different chipsets.  

Easiest place to start, try to replicate the paper. They have 3 criteria:  
1. Complete failure to run: If the function does not run on the device at all.  
2. Partial failure to run: Some but not all the benchmark tests for a given function fail to run.  
3. Intolerable latency: High latencies may be prohibitively inefficient, which may impair usability even if the function technically is able to run on multiple hardware types. (*note: 'intolerable' may very well be different per function, may need to have an intolerable.txt file that determines the value for each function.)  
  
To start with: JAX version 0.4.8, PyTorch version 1.12.0, and TensorFlow version 2.11.0.  
  
In the paper, they do some data analysis to find the best functions to test. As a dumb first approximation, we can just test against the functions they list. Obviously, future improvements would need to re-analyze to find the appropriate functions for each version.  

#PyTorch  
torch.argsort   
torch.optim.Adamax   
torch.fliplr  
torch.broadcast_tensors  
torch.nn.AdaptiveAvgPool3d   
torch.addr  
torch.cat  
torch.optim.LBGFS  
torch.triangular_solve  
torch.nn.Module.state_dict  
torch.nn.Module.zero_grad  
torch.sum  
torch.Tensor.is_same_size  
torch.nn.KLDivLoss  
torch.nn.LSTMCell  
torch.moveaxis  
torch.nn.functional.dropout  
torch.lt  
torch.autograd.Variable  
torch.utils.data.Subset  
torch.nn.Sequential  
torch.multinomial  
torch.nn.Linear  
torch.nn.BCEWithLogitsLoss  
torch.diag  
torch.von_mises.VonMises  
torch.zeros  
torch.round  
torch.range  
torch.autograd.functional.jvp  
torch.linalg.matrix_rank  
torch.nn.GELU  
torch.Tensor.to  
torch.nn.Transformer  
torch.bitwise_not  
torch.nn.Conv3d  
torch.slogdet  
torch.optim.lr_scheduler.ExponentialLR  
torch.utils.data.Dataset  
torch.utils.data.ConcatDataset  
torch.nn.Parameter.register_parameter  
torch.cuda  
torch.nn.Conv2d  
