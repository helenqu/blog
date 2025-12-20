---
title:  "RL's Deadly Triad Meets Optimization"
author: "Helen Qu"
date: 2025-12-19
math: true
tags: ["rl", "optimization"]
draft: true
---

{{< macros >}}
\newcommand{\phit}{\phi_t}
\newcommand{\phinext}{\phi_{t+1}}
\newcommand{\enext}{\ebf_{s_{t+1}}}
\newcommand{\et}{\ebf_{s_t}}
\newcommand{\gradV}{\nabla_{\theta} V_{\theta}}
\newcommand{\V}{V_{\theta}}
\newcommand{\snext}{s_{t+1}}
\newcommand{\tstar}{\theta^\star}
\newcommand{\itmtx}{(\Ibf - \eta \Abf)}
\newcommand{\AMC}{\Abf_{\text{MC}}}
\newcommand{\Aon}{\Abf_{\text{on}}}
\newcommand{\Atab}{\Abf_{\text{tab}}}
\newcommand{\dpi}{d_{\pi}}
\newcommand{\Ppi}{P_{\pi}}
\newcommand{\Tpi}{\Tcal_{\pi}}
{{< /macros >}}

<!-- Baird's counterexample is a canonical concrete system that demonstrates the instability of semi-gradient Q-learning in the presence of the so-called "deadly triad" of reinforcement learning: functional approximation, off-policy learning, and bootstrapping. -->
At this point, the so-called "deadly triad" of reinforcement learning has a status approaching that of folklore.
This unsavory combination of **function approximation, off-policy learning, and bootstrapping** leads to unstable learning dynamics, even in the simplest of systems.
Most RL enthusiasts are familiar with it, yet I've seen surprisingly little discussion on its origins or first principles.
I made this post to shed light on the deadly triad from the perspective of traditional optimization theory, with the goal of demonstrating that the underlying principles are nothing more exotic than well-established convergence guarantees for dynamical systems.

## Why adopt Deadly Triad learning paradigms?

If the deadly triad is so deadly, why not avoid it altogether? 
- **Function approximation** (e.g., via neural networks) allows us to parameterize very high-dimensional state-action spaces that would otherwise fall prey to the curse of dimensionality (e.g., [DQN](https://arxiv.org/abs/1509.06461)). Of course, real-world state-action spaces (e.g., multimodal video + sensor inputs in self-driving) are fundamentally high-dimensional so this problem isn't going away.
- **Off-policy learning** is a convenient way to disentangle the data source (behavior policy) from the target policy. Since the optimal policy for data generation will rarely coincide with the globally optimal policy due to the need for exploration, this allows the data generation policy to decouple from the learned optimal target policy. Additionally, true on-policy learning is difficult to implement in practice at scale in distributed setups.
- Finally, **bootstrapping** replaces the true return at state $s_t$, which requires rolling out a full trajectory to calculate, with a one-step rollout ($r_t$) and approximates the rest with the bootstrapped value function estimate $\V(\snext)$. This is a straightforward way to balance the trade-off between computational cost and accuracy.

## Setup

<!-- Baird's counterexample is a simple Markov decision process with 7 states: one "upper" state and six "lower states". An agent at any lower state transitions deterministically to the upper state, while an agent at the upper state will transition at random to one of the six lower states. There are no rewards in this system, so the true value function $V^{\star}$ is identically 0 for all states. -->
We adopt the learning setup of [**Baird's counterexample**](http://leemon.com/papers/1995b2.pdf), an embarrassingly simple concrete system (with no rewards!) that demonstrates the instability of semi-gradient Q-learning in the presence of the deadly triad. I'll introduce the relevant ingredients here.

For simplicity, the value function is parameterized with a linear model $\V(s) = \transpose{\phi(s)} \theta$, where $\phi(s)$ is a feature vector corresponding to state $s$ and $\theta$ are the learned feature weights, which we wish to optimize with temporal difference (TD(0)) learning.

The key result in Baird's counterexample is that learning diverges, even for such a conceptually simple system with no rewards, due to the deadly triad:
* **Function approximation**: We use a linear parameterization for the value function $\V$.
* **Boostrapping**: The TD family of methods uses bootstrapping to iteratively refine value function estimates (as opposed to, e.g., Monte Carlo estimation).
* **Off-policy learning**: Q-learning is an off-policy variant of TD learning 

## The TD Algorithm
<!-- how much context do i give??? -->

The TD(0) update rule at step $t+1$ for weights $\theta$ takes the form $\theta_{t+1} = \theta_t + \eta \delta_t \gradV$, where $\delta_t = r_t + \gamma \V(\snext) - \V(s_t)$ represents the current approximation error between $\V$ and $V^{\star}$ ($\eta$ is the learning rate, $\gamma$ is the discount factor for future rewards, and $r_t$ is the reward at state $s_t$). Our linear model for $\V$ gives $\gradV = \phi(s_t)$, so the expected update can be written as

\\[ 
    \begin{aligned}
    \Delta \theta &= \ev[d_b]{\theta_{t+1} - \theta_t} \cr
                  &= \ev[d_b]{\eta ( r_t + \gamma \transpose{\phinext} \theta_t - \transpose{\phi_t} \theta_t ) \phi_t} \cr
                  &= \ev[d_b]{\eta (r_t \phi_t - \phi_t(\transpose{\phi_t} - \gamma \transpose{\phinext}) \theta_t)}
    \end{aligned}
\\]
where we define $\phi_i \equiv \phi(s_i)$ for simplicity, and $d_b$ is the distribution over states drawn from the behavior policy $b$.

We can define 
\\[
    \Abf \equiv \ev[d_b]{\phi_t(\transpose{\phi_t} - \gamma \transpose{\phinext})}, \cbf \equiv \ev[d_b]{r_t \phi_t}
\\]
to see that the above expected update is of the form $\Delta \theta = \eta (\cbf - \Abf \theta)$.

## Insights from Optimization Theory

An update of this form corresponds to a system where convergence depends entirely on the spectral radius $\rho$ of the matrix $\itmtx$. A relatively easy way to see this intuitively is as follows:

If it exists, the fixed point of this system (i.e., a point where $\Delta \theta = 0$) is $\tstar = \Abf^{-1}\cbf$.
We define the error $e_t$ at iteration $t$ as 
\\[
    e_t \equiv \theta_t - \tstar
\\]
which evolves as $e_{t+1} = \itmtx e_t \Rightarrow e_t = \itmtx^t e_0$.[^TDerr] We can define convergence as $\lim_{t \to \infty} \norm{e_t} = 0$, meaning that we require 
\\[
    \lim_{t \to \infty} \itmtx^t = 0 \Rightarrow \rho \itmtx < 1
\\]
to guarantee convergence.[^specradius_convergence]

If $\Abf$ is symmetric positive definite (SPD, i.e., all eigenvalues of $\Abf$ are real and positive), we know how to compute learning rates $\eta > 0$ for which convergence is guaranteed. We note that this is sufficient but not necessary for convergence, but $\Abf$ not SPD indicates that the convergence guarantee is broken. The central thesis of this post rests on this statement, and much of it will be dedicated to understanding the deadly triad in the context of the properties of $\Abf$.

<!-- # Convergence guarantee does not hold in Baird's Counterexample -->

<!-- We show now that convergence is not guaranteed in Baird's counterexample by demonstrating that $\Abf = \ev[d_b]{\phi_t(\transpose{\phi_t} - \gamma \transpose{\phinext})}$ is not SPD.
First, we rewrite $\Abf$ in matrix form:
\\[
    \Abf = \transpose{\Phi} D_b (\Ibf - \gamma P_{\pi}) \Phi
\\]
where:
- columns of $\Phi$ correspond to $\phi(s)$ for all states $s$ in the system
- $D_b \equiv \text{diag}(d_b(s))$ is a diagonal matrix of probabilities of state $s$ under the behavior policy $b$
- $P_{\pi} \equiv \sum_{a} \pi(a \mid s_t) P(\snext \mid s_t,a) $ is the transition matrix under the target policy $\pi$ -->

<!-- An SPD matrix $\Mbf$ satisfies the relation $\transpose{x} \Mbf x > 0 \; \forall x$, so we must show that there exists $x \neq 0$ for which $\transpose{x} \Abf x \leq 0$.

To construct such an $x$ we look at the properties of the Baird's counterexample MDP.

[FILL THIS IN] -->

## The Deadly Triad
What can this lens of optimization bring to our understanding of the deadly triad more generally? We ablate each of the components in this section and demonstrate mathematically how, in all cases, convergence guarantees hold (either through demonstrating that $\Abf$ is SPD or otherwise).

### No bootstrapping

We replace TD learning with Monte Carlo estimation to expose the role of bootstrapping in the deadly triad.  Instead of estimating the true return $G_t$ with bootstrapped value function estimates ($G_t \approx r_t + \gamma \V(\snext)$), Monte Carlo estimation uses the true discounted return at time $t$, $G_t \equiv r_t + \gamma r_{t+1} + ... + \gamma^{T-t-1} r_{T-1}$, where $T$ represents the number of timesteps of an episode/trajectory. The trade-off of MC estimation with TD learning is simply cost, since $G_t$ is computed by rolling out a policy through a full trajectory.

The MC estimation expected update is 
\\[ 
    \begin{aligned}
    \Delta \theta &= \ev[d_b]{\eta ( G_t - \V(s_t) ) \gradV} \cr
                  &= \ev[d_b]{\eta ( G_t - \transpose{\phi_t}\theta ) \phi_t}
    \end{aligned}
\\]

We can similarly write this into $\Delta \theta = \eta (\cbf - \Abf \theta)$ form by defining 
\\[
    \AMC \equiv \ev[d_b]{\phi_t \transpose{\phi_t}}, \cbf_{\text{MC}} \equiv G_t \phi_t
\\]
Similar to the previous derivation, we write $\Abf_{\text{MC}}$ in matrix form:
\\[
    \AMC = \transpose{\Phi} D_b \Phi
\\]
Comparing $\AMC$ with $\Abf$, we can see that removing bootstrapping has the direct effect of replacing $\Ibf - \gamma P_{\pi}$ with simply $\Ibf$. Intuitively, we use the next state $\snext$ (drawn from $P_{\pi}$) in TD learning to approximate the expected return, but this is replaced by the true return $G_t$ in MC estimation.

We can prove that $\AMC$ is SPD. We define $\xbf \in \text{span}(\AMC) \neq 0$ and show that $\transpose{\xbf} \AMC \ubf > 0$:
\\[
    \begin{aligned}
  \transpose{\xbf} \AMC \ubf &= \transpose{\xbf} \transpose{\Phi} D_b \Phi \xbf \cr
  &= \sum_{s_t} d_b(s_t)(\transpose{\phi_t} \xbf)^2 \cr
  &> 0
    \end{aligned}
\\]
as long as $d_b(s_t) > 0$ (true unless state $s_t$ is never visited under $b$) and $\phi_t$ has full column rank.

This shows that learning converges for any MDP with linear value function approximation and off-policy learning as long as the true return is calculated rather than estimated via bootstrapping.

### On-policy

Replacing off-policy with on-policy learning intuitively replaces the behavior policy's distribution of states $D_b$ with that of the target policy $D_{\pi}$ (since they are now one and the same):
\\[
    D_b = D_\pi \Rightarrow \Aon \equiv \transpose{\Phi} D_{\pi} (I - \gamma P_{\pi}) \Phi
\\]

We now want to show that $\Aon$ is SPD, defining $\xbf \in \text{span}(\AMC) \neq 0$ the same way as above and additionally defining $\ubf = \Phi \xbf$ for convenience.
Now we can write
\\[
    \begin{aligned}
    \transpose{\xbf} \Aon \xbf &= \transpose{\ubf} D_{\pi} (I - \gamma P_{\pi}) \ubf \cr
    &= \inner[D_{\pi}]{\ubf}{\ubf} - \gamma \inner[D_{\pi}]{\ubf}{P_{\pi} \ubf} \cr
    & \leq  \norm[D_{\pi}]{\ubf}^2 - \norm[D_{\pi}]{P_{\pi} \ubf} \norm[D_{\pi}]{\ubf} 
    \end{aligned}
\\]
where the final line follows by the Cauchy-Schwarz inequality ($\inner[D_{\pi}]{\ubf}{P_{\pi} \ubf} \leq \norm[D_{\pi}]{P_{\pi} \ubf} \norm[D_{\pi}]{\ubf}$). This means that in order for $\Aon$ to be SPD (i.e., $\transpose{\xbf} \Aon \xbf > 0$), we have to show $\norm[D_{\pi}]{P_{\pi}\ubf} \leq \norm[D_{\pi}]{\ubf}$.

We can write $\norm[D_{\pi}]{P_{\pi}\ubf}$ as an expected value and apply Jensen's equality since $x^2$ is a convex function of $x$:
\\[
    \begin{aligned}
    \norm[D_{\pi}]{P_{\pi}\ubf}^2 &= \sum_s d_{\pi}(s) ( \ev[P_{\pi}]{\ubf_{s'}} )^2 \cr
    & \leq  \sum_s d_{\pi}(s) \ev[P_{\pi}]{\ubf_{s'}^2} = \sum_s d_{\pi}(s) \sum_{s'} P_{\pi}(s' \mid s) \ubf_{s'}^2 \cr
    &= \sum_{s'} \ubf_{s'}^2  \sum_s d_{\pi}(s) P_{\pi}(s' \mid s)
    \end{aligned}    
\\]
Since $d_{\pi}$ and $P_{\pi}$ are defined for the same policy $\pi$, $d_{\pi}$ is *stationary* for $P_{\pi}$: $\transpose{\dpi} \Ppi = \transpose{\dpi}$. This gives us
\\[
    \sum_s d_{\pi}(s) P_{\pi}(s' \mid s) = \transpose{\dpi} \Ppi = \transpose{\dpi} = \sum_{s'} \dpi(s')
\\]
Using this in the previous equation block, we see that
\\[
    \begin{aligned}
    \norm[D_{\pi}]{P_{\pi}\ubf}^2 &\leq \sum_{s'} \ubf_{s'}^2  \sum_s d_{\pi}(s) P_{\pi}(s' \mid s) \cr
    &= \sum_{s'} \ubf_{s'}^2 \dpi(s') \cr
    &= \norm[D_{\pi}]{\ubf}^2
    \end{aligned}
\\]

This means that 
\\[
    \norm[D_{\pi}]{\ubf}^2 \geq \norm[D_{\pi}]{P_{\pi} \ubf} \norm[D_{\pi}]{\ubf} 
    \Rightarrow \norm[D_{\pi}]{\ubf}^2 - \norm[D_{\pi}]{P_{\pi} \ubf} \norm[D_{\pi}]{\ubf} \geq 0
\\]
Recall that this expression was our lower bound for $\transpose{\xbf} \Aon \xbf$, concluding our proof that $\Aon$ is SPD.

Remember that this proof hinges on the fact that $\dpi$ is stationary with respect to $\Ppi$. In off-policy learning, $D_{\pi}$ would be replaced by $D_b$, since we draw states/actions from the behavior policy. It is *not true in general* that $d_b$ is stationary for $\Ppi$: this is the crux of the role of off-policy learning in the deadly triad.

### No function approximation

The alternative to function approximation is a tabular representation of the value function: a discrete mapping between states and estimated values. The convergence story here relies on the fact that TD learning is now acting on the state-space itself (rather than its projection through $\phi$ into some feature space). 

We start with the general expected TD(0) update at iteration $k+1$:
\\[
    V_{k+1}(s_t)= \ev[d_b]{V_k(s_t) + \eta (r_t + \gamma V_k(\snext) - V_k(s_t))}
\\]
<!-- and define the expected update as 
\\[
    \begin{aligned}
    \Delta V &= \ev[d_b]{V_{k+1}(s_t) - V_k(s_t)} \cr
            &= \ev[d_b]{\eta (r_t + \gamma V_k(s_{t+1}) - V_k(s_t))}
    \end{aligned} 
\\] -->
Recall that the Bellman operator $\Tpi$ for policy $\pi$ is defined as 
\\[
    (\Tpi V_k)(s) = \ev[d_{\pi}]{r_t + \gamma V_k(s_{t+1}) \mid s_t = s}
\\]
Now we see that the expected TD(0) update can be written simply in terms of $\Tpi$ [^impsamp]:
\\[
    V_{k+1}(s) = V_k(s) + \eta ((\Tpi V_k)(s) - V_k(s))
\\]
Since the Bellman operator is a contractive mapping ($\norm[\infty]{\Tpi v - \Tpi w} \leq \gamma \norm[\infty]{v - w}$), we can easily show the same contractive property for the update mapping $\text{TD}(v) \equiv v + \eta (\Tpi v - v)$:
\\[
    \begin{aligned}
    \norm[\infty]{\text{TD}(v) - \text{TD}(w)} &= \norm[\infty]{(1-\eta)(v-w) + \eta(\Tpi v - \Tpi w)} \cr
    &\leq ((1-\eta) + \eta \gamma) \norm[\infty]{v - w} \cr
    &= (1-\eta(1+\gamma)) \norm[\infty]{v - w}
    \end{aligned}
\\]
As long as $(1-\eta(1+\gamma)) < 1$, the TD update for tabular value functions is a contraction and thus guaranteed to converge.
 
## Conclusion

We've seen from an optimization perspective that each component of the deadly triad brings its own unique source of instability, and that removing each component individually leads to a converging system. The key to employing algorithms that have deadly triad properties then is to identify and alleviate/eliminate these underlying sources of instability.

---
Acknowledgements here

[^specradius_convergence]: This implication was not directly obvious for me, so I'll write out a quick proof. We know $\rho(\Mbf) = \max_{\lambda} \abs{\lambda}$, and for any eigenvalue $\lambda$ of matrix $\Mbf$ with associated eigenvector $\vbf$, $\Mbf \vbf = \lambda \vbf$ and $\Mbf^t \vbf = \lambda^t \vbf$. Thus, $\Mbf^t \vbf \rightarrow 0 \Rightarrow \lambda^t \rightarrow 0$. This directly implies $\max(\lambda) < 1 \Rightarrow \rho(\Mbf) < 1$.
[^TDerr]: Quick derivation of $e_{t+1} = (\Ibf - \eta \Abf)e_t$: 

    $$
        \begin{aligned}
        e_{t+1} &= \theta_{t+1} - \tstar \cr
        &= \theta_t + \eta (\cbf - \Abf \theta_t) - \tstar \cr
        &= \eta \Abf \tstar + (\Ibf - \eta \Abf) \theta_t - \tstar \cr
        &= (\Ibf - \eta \Abf)(\theta_t - \tstar) \cr
        &= (\Ibf - \eta \Abf) e_t
        \end{aligned}
    $$

[^impsamp]: We note that when target policy $\pi$ and behavior policy $b$ have very different distributions over actions / states, importance sampling with weighting $\pi(a_t \mid s_t) \over b(a_t \mid s_t)$ must be used to equate $\ev[d_b]{\cdot}$ with $\ev[d_{\pi}]{\cdot}$.
<!-- https://www.reddit.com/r/reinforcementlearning/comments/17cm8bg/is_the_deadly_triad_even_real/ -->