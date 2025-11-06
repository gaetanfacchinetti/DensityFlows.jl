# Overview

DensityFlows.jl is a lightweight Julia package for data scientists and physicists who want a simple way to model low-dimensional probability distributions using normalizing flows. It’s built for clarity and ease of use — ideal for anyone who wants to experiment, learn, or prototype quickly without the overhead of large ML frameworks. While other libraries focus on complex, high-dimensional tasks like image generation, DensityFlows.jl keeps things minimal and transparent, helping you understand and apply normalizing flows right away.


## Basics of normalizing flows

Let us say we want to emulate a conditional probability $P$ with distribution function $p(x \, |\, \theta)$ (equivalent to a likelihood) for $x\in \mathcal{D} \subset \mathbb{R}^d, \theta \in \mathcal{E} \subset \mathbb{R}^n$, with $d \in \mathbb{N}_*$ and $n \in \mathbb{N}$. To that end we can start from a probability distribution function $Q$ with distribution function $q$ that is known and perform a change of variable from $q$ to $p$. In practice, we thus want to find the diffeomorphism $f_\theta$ that, for $z \sim Q$ satisfies $f_\theta(z) \sim P$. This requirement imposes that $f_\theta$ satisfies

```math
p(x \, |\, \theta) = q(f_\theta^{-1}(x)) \left| {\rm det} \,  J[f_\theta^{-1}](x) \right| \quad \forall (x, \theta) \in \mathcal{D} \times \mathcal{E}
```
with $J[f_\theta^{-1}]$ the Jacobian of the inverse transformation. Moreover, using the properties of the Jacobian, this can also be written
```math
p(x \, |\, \theta) = \frac{q(f_\theta^{-1}(x))}{\left| {\rm det} \,  J[f_\theta](f^{-1}_\theta(x)) \right|} \quad \forall (x, \theta) \in \mathcal{D} \times \mathcal{E}
```

Now, let us assume that $f_\theta$ is written as a composition of $m$ elementary diffeomorphisms as follows
```math
f_\theta = g_{\theta, m} \circ g_{\theta, m-1} \circ \dots \circ g_{\theta, 1}.
```
These diffeomorphisms can be defined using neural networks. Then, using the chain rule, for $\theta \in \mathcal{E}$ and $z \in f_\theta^{-1}(\mathcal{D})$, 
```math
\begin{equation*}
\begin{split}
J[f_\theta](z) & = J[g_{\theta, m} \circ g_{\theta, m-1} \circ \dots \circ  g_{\theta, 3}\circ g_{\theta, 2} \circ g_{\theta, 1}](z) \\
 & = J[g_{\theta, m} \circ g_{\theta, m-1} \circ \dots \circ g_{\theta, 3} \circ g_{\theta, 2}](g_{\theta, 1} (z)) \times  J[g_{\theta, 1}](z) \\
  & = J[g_{\theta, m} \circ g_{\theta, m-1} \circ \dots \circ g_{\theta, 3}](g_{\theta, 2} \circ g_{\theta, 1}(z)) \times  J[g_{\theta, 2}](g_{\theta, 1} (z)) \times  J[g_{\theta, 1}](z) \\
 & = \dots \\
 & = \prod_{i=1}^{m} J[g_{\theta, i}](g_{\theta, i-1} \circ \dots \circ g_{\theta, 1}(z)) \, .
\end{split}
\end{equation*}
```
Let us now apply this relationship to $z = f_\theta^{-1}(x)$ where $x\in \mathcal{D}$,
```math
\begin{equation*}
\begin{split}
J[f_\theta](f^{-1}_\theta(x)) & = \prod_{i=1}^{m} J[g_{\theta, i}](g_{\theta, i-1} \circ \dots \circ g_{\theta, 1} \circ g_{\theta, 1}^{-1}  \circ \dots \circ g_{\theta, m}^{-1} (x))\\
& = \prod_{i=1}^{m} J[g_{\theta, i}](g_{\theta, i}^{-1}  \circ \dots \circ g_{\theta, m}^{-1} (x)) \, .
\end{split}
\end{equation*}
```
and therefore
```math
{\rm det} \,  J[f_\theta](f^{-1}_\theta(x))  = \prod_{i=1}^{m} {\rm det} \, J[g_{\theta, i}](g_{\theta, i}^{-1}  \circ \dots \circ g_{\theta, m}^{-1} (x)) 
```
In other words, we have shown that the determinant of the Jacobian can be computed recursively by multiplying the Jacobian of every diffeomorphism evaluated at a point that only depends on the previous inverse diffeomorphisms. In practice for a large number of dimension the jacobian can be long to evaluate. One solution is to use transformation with triangular jacobians which can be computed much faster.

## Loss function

The loss function associated to the determination of $f_\theta$ is the Kullback-Leibler divergence between $p$ and the _sampled_ distribution $r$
```math
\begin{equation*}
\begin{split}
L & = \int  r(x \, | \, \theta) \ln\frac{r(x \, | \, \theta)}{p(x \, | \, \theta)} \, {\rm d} x \\
& = - \mathbb{E}_r \left[\ln p(x \, | \, \theta) \, | \, \theta \right] + {\rm cst.} \\
& = - \mathbb{E}_r \left[\ln q(f_\theta^{-1}(x)) - \ln \left| {\rm det} \,  J[f_\theta](f^{-1}_\theta(x)) \right| \, | \, \theta\right]\, .
\end{split}
\end{equation*}
```
For a sample of $N$ points $\{(x, \theta)_i\}_{i\in [1, N]}$ it can be estimated as
```math
\begin{equation*}
\begin{split}
L & \simeq \frac{1}{N} \sum_{i = 1}^N \left[\ln q(f_{\theta_i}^{-1}(x_i)) - \sum_{j=1}^m \ln \left| {\rm det} \,  J[g_{\theta_i, j}](g_{\theta_i, j}^{-1}  \circ \dots \circ g_{\theta_i, m}^{-1} (x))  \right| \right] \, .
\end{split}
\end{equation*}
```


## Layers

Currently, only NICE and RNVP coupling layers are implemented.

For more information see
- [Laurent Dinh, David Krueger, Yoshua Bengio (2014). NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516).
- [L. Dinh, J. Sohl-Dickstein, S. Bengio (2017). Density Estimation Using Real NVP. International Conference on Learning Representations.](https://arxiv.org/abs/1605.08803).


## From here and beyond

In order to get familiar with the code please give a look at the [documentation](./documentation.md) and at the detailed [example](./example.md). Any comment to the code or suggestion for improvement is welcome, please do so using the GitHub issues page if relevant. Further developement should include the implementation of conditional masked autoregressive flows (CMAF).