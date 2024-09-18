# MudJack  
  
This is a testing framework for ML libraries. Intended targets: PyTorch, TensorFlow and JAX.  
  
The inspiration for this testing benchmark is this paper by Mince, Dinh, Kgomo, Thompson and Hooker (The Grand Illusion: The Myth of Software Portability and Implications for ML Progress)[https://arxiv.org/abs/2309.07181]. Apparently ML libraries do not work across different chipset architectures.  
  
This code is intended to be run against all of these frameworks and report on errors and performance differences.  
  
The planned architecture:  
1. Basic tests of functionality in all 3 languages, 3 different runtimes.  
2. These will be spun up in separate Container runtimes with an api endpoint.  
3. On calling the api endpoint, they will return a file (probably json, maybe csv?) with the results of the test.  
4. Using a Kubernetes framework and correctly managed nodes with the appropriate chips, we should be able to run this test across different chipsets.  
