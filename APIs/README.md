## Generator Code Snippets

This folder contains different implementations of NumPy APIs which are used to create a dataset of graphs which can be used to train our ML model. 
The implementations are written in a way that they only use pure Python code, they do not call any APIs which may be written in a different language. 
This is done so that our framework can handle the implementations, as any code not written in pure Python would end up being a black box.
The implementations are also written in a way that they handle all possible behaviors of the NumPy API they represent, so that one implementation is able to represent all possible behavior of an API.
