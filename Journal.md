# Thesis Journal logging
## Erik Lager


### 5 April 2023
Getting a value of infinity makes sense because if we look at the easiest case with transporting (1,-1) to (-1,1), where we expect the transport value to be 2, we can see that because the distance from an element to itself is zero, which means the distance matrix looks like:
|   | t1 | t2 |
| - | - | - |
| s1 | 0 | 1 |
| s2 | 1 | 0 |
This means that for any k &in; N, the transport plan can be :
|   | t1 | t2 |
| - | - | - |
| s1 | 1-k | k |
| s2 | k-2 | 1-k |
This results in a cost function of K+K-2=2K-2. We can easily notice that non-negativity is impossible for this transport matrix. When we remove such a constraint, we could make K as high or as low as we would like and get a solution anywhere between - &infin; and + &infin;


### 6 April 2023
*Explaining Tobias's work*
The functions are related to solving unbalanced optimal transport problems using the iterative scaling algorithm. The goal is to find the optimal mapping between two distributions with a given cost matrix.

**div0** and **mul0** are helper functions used to avoid zero division and multiplication with infinity, respectively.

**proxdiv** is the proximal operator of the divergence function F. It takes in the current state variables and updates them according to the chosen divergence function.

**fdiv** calculates the value of the divergence function F given the current variables and parameters.

**fdiv_c** calculates the convex conjugate of the divergence function F given the current variables and parameters.

**simple_scalingAlg** is an implementation of the iterative scaling algorithm for solving unbalanced optimal transport problems.

**make_1D_gauss** returns a 1D histogram for a Gaussian distribution.

**plot1D_mat** plots the matrix with the source and target 1D distribution.

**plot2D_samples_mat** plots a matrix in 2D with lines using alpha values.

**full_scalingAlg** is an implementation of the iterative scaling algorithm that includes the log-domain stabilization.

**signed_GWD** is a function for solving the signed Gromov-Wasserstein discrepancy problem using the iterative scaling algorithm.

### 9 April 2023
Understood and got same results for both full_scalingAlg and ot.sinkhorn, now I need to compare everything to how it is used in the old code. 


### 11 April 2023
Been strugling with second test for a couple of days now, apparently, using a "too small" regularization parameter, made the sinkhorn go crazy and only perform one iteration(and fail) which led to a trivial transport plan. Thus meaning that there should be something to test if the regularization parameter is too small.


### 19 April 2023
Been reading https://arxiv.org/pdf/2211.08775.pdf for the last few days, mainly focusing on chapters 3 and 4. The way they did unbalanced sinkhorn iterations in 4 was using the algorithm in page 31, this is considered to be the more stable way if we compare to matrice multiplication(the iterations I tried to do so far) but it is *MUCH* slower. 
For graph drawing, I used tobias' code and functions. 
To go over some of his code:

Test1 took a balanced case and applied the KL constraint.
Test2 took a balanced case and applied the TV constraint.

Both tests were not entirely correct since we could see that the Transport plan did not sum up to 1. I implemented both using the functions from the pot library and got feasible transport plans. 

Though, I did not get the clear graphs he got in the "Not transported mass" and "Marginals of transport plan vs target and source distributions"


Some TODO's:
watch https://www.youtube.com/watch?v=JT99qBSsiJo&list=WL&index=18&t=145s , which includes marco, peyre, and some other researchers talking about OT applications in late 2022.

Try and understand how to regularize for negative distances, haven't meditated enough over what Tobias did, and I've seen other ways in other papers. Should check them all out and see what makes sense. 

The scaling iterations are a little different between what tobias used and what they used in the recent paper, I haven't gone too deep enough to understand that difference.














