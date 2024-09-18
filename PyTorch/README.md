pip install fastapi uvicorn torch==1.12.0

curl http://127.0.0.1:8000/test

Notes  
Deprecations and Adjustments:  
  
torch.range: Deprecated in favor of torch.arange. The code uses torch.arange.  
torch.autograd.Variable: Deprecated but still available for backward compatibility.  
torch.Tensor.is_same_size: Replaced with comparing tensor sizes.  
torch.optim.LBGFS: Corrected to torch.optim.LBFGS.  
torch.von_mises.VonMises: Adjusted to torch.distributions.VonMises.  
Error Handling: If a function raises an exception, the error traceback is captured and included in the response.  
  
Timing: Execution time is measured using time.time() and averaged over 10 runs to provide a stable estimate.  
