Project Generalized OT geodesics

The goal is to make a lagrangian code using pytorch to approximate geodesics between two distribution for generalized Wasserstein distances

We denote X0 and X1 two (d,n) matrices of n point in dimension d. The code should handle a generic d and we write an illustrative jupyter .ipynb for dimension 2.
We aim at optimizing with L-BGFS the energy of X in (d,n,T) where T is the number of time stepping an energy
    E(X) := W_2^2(X0,X_T) + W2(X0,X_0) + gamma * sum_{t=1}^p sum_{i=1^1} phi(X_t,X_t(i)) |X_t(i)-X_{t-1}(i)|^2
where X_t(i) is the ith point at instant t in X, i.e. X[:,i,t] and phi a generic function. 

For classical OT, it is just phi=1 (constant), and for transformers-ot, it should be independent of x and 
    phi(X,x) := sum_{i,j} exp(-|X_i-X_j|^2)

Here W_2^2 is the squared wasserstein distance, ie
    W_2^2(Y,Z) := inf_{sigma in Perm(n)} |Y_i-Z_{sigma(i)}|^2
where Perm(n) is the set of permutaiton of n point. You should compute W_2^2 using POT python library.


You should write a nice pytorch-based library, a nice github-ready repository, with e g code/ example/ paper/, with function codeded and documented in a .py main file, with helper function in a helper.py separate file, so that notebook of example import them, either running locally or on colab (so check and do the correct path adaptation).  

Summerize the maths and important thing in a latex document in paper/.

Make a jupyter example, where X0 and X1 are two gaussian, gamma is large, and you initialize X by just interrly interpolating X0 and X1. Visualize for several step of the optimization the obtained interpolation X between X0 and X1.