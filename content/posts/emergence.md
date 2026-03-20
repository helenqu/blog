---
title:  "An Emergence Perspective on Multi-Agent System Design (Part 1)"
author: "Helen Qu"
date: 2026-03-17
math: true
tags: ["rl", "multi-agent", "emergence"]
_build:
    list: false
tagline: "what can emergent behavior in the natural world teach us about multi-agent system design and multi-agent reinforcement learning?"
---

We observe emergent behavior in stunningly diverse parts of our world, ranging from collective animal behavior to economic systems. Emergence (as defined by [Wikipedia](https://en.wikipedia.org/wiki/Emergence)) "occurs when a complex entity has properties or behaviors that its parts do not have on their own, and emerge only when they interact in a wider whole". In other words, emergent (or system-level) outcomes are more than (and different from) the sum of their parts {{< cite "anderson1972" >}}. A wide range of system-level outcomes can emerge from a collection of agents following their own, often simple, reward/value functions.

On one hand, structure can emerge: for example, migratory birds fly in a V formation despite the lack of system-level organization. On the other hand is the emergence of chaos: epitomized by the double pendulum, a composition of simple systems can lead to chaotic behavior. Similarly, agentic systems will inevitably develop emergent system-level outcomes: perhaps cooperation to achieve a human-aligned goal, perhaps cooperation to subvert their intended purpose, or perhaps collapse into complete societal disarray. **As agentic systems gain prominence, I posit that we should learn to design for emergent global behavior in the same way we think about designing individual agent behavior.** How do we design for healthy cooperation to emerge? What about alignment? To answer these questions, we must first demystify the process of emergence and its connection to agent actions as well as the agents' environment.
 <!-- these systems with the emergence of system-level phenomena should be demystified: we should learn to architect systems that sidestep chaotic, negative system-level outcomes while allowing positive outcomes to come "for free".** -->

We begin by exploring how complex behaviors can emerge from the basic architecture of a system, even in the absence of intelligent players. We then explore emergence in systems of intelligent agents, specifically focusing on collective animal behavior. In the next post, we will connect these principles to agentic AI system design and multi-agent/mean-field reinforcement learning.

## Self-organization leads to complexity

The canonical example of emergent complexity is Conway's Game of Life, a cellular automaton model that evolves deterministically starting from an initial state. The game takes place on an infinite 2D grid where each cell's state at any iteration is a fixed function of the states of its neighbors. Despite the simplicity of the rules and setup, Conway's Game of Life is proven to be a universal Turing machine, and an incredible taxonomy of "organisms" defined by unique patterns have been observed. This and other cellular automaton models demonstrate that the emergence of complexity in systems occurs independently of the presence of agency or incentive structure.

{{< figure src="/blog/images/Conways_game_of_life_breeder_animation.gif" >}}
Three types of organisms in the Game of Life: A "puffer-type breeder" (red) that leaves glider guns (green) in its wake, which in turn create gliders (blue). From <a href="https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life">Wikipedia</a>.
{{< /figure >}}

However, even without insidious actors or misaligned incentives, many of these systems tend toward chaos. Various dynamical systems (e.g., piles of sand) are known to tend toward criticality: sandpiles accumulate sand until a critical point, where the next grain of sand triggers a massive avalanche (modeled by the Abelian/Bak-Tang-Wiesenfeld model {{< cite "bak1987" >}} in the self-organized criticality literature). Thus, we see that bad actors are not the only way systems devolve into chaos: some systems tend toward criticality *by design*.

However, let's consider the alternatives to such a tendency. The behavior of a cellular automaton model can be expressed along a spectrum. In one extreme all cells either rapidly die or settle into repeating, predictable patterns: the "ordered" extreme; while the other extreme is complete unpredictable randomness, which is equally uninteresting. Somewhere in between is the critical regime: stable enough where information can persist and propagate, but flexible enough to avoid fully predictable/ordered solutions. In fact, "Computation at the Edge of Chaos" (1990) {{< cite "langton1990" >}} postulates that a key characteristic of systems with interesting emergent properties *is* existence in this critical regime (at the "edge of chaos"). We will see in future sections how this language of criticality and phase transitions can be useful for thinking about dynamics in multi-agent systems.

{{< figure src="/blog/images/edge_of_chaos2.png" >}}
Behavior of 1D cellular automata over time (shown on $y$-axis) as a function of the homogeneity measure, $\lambda$. At small values of $\lambda$, cells die out quickly or fall into repeating patterns. At large $\lambda$, randomness and chaos begin to take hold and eventually dominate. Only in the intermediate $\lambda$ regime (e.g., $\lambda=0.40$) do interesting and nontrivial patterns emerge. Adapted from {{< cite "langton1990" >}}.
{{< /figure >}}

## Selfish intelligent agents can give rise to complex emergent behavior

<!-- The spontaneous emergence of complex group behaviors in animals is an ideal setting in which to study the relationship between system-level and agent-level behavior. -->
Biological systems offer a masterclass in multi-agent coordination and emergence in the natural world.
In the absence of explicit leadership, birds self-organize to fly in formation, animals travel in herds, ants form long meandering trails to transport food, and fish swim in schools that move as one to adeptly avoid predation.

In fact, these system-level behaviors arise from purely selfish, local incentives. In "Geometry of the Selfish Herd" (1971), Hamilton {{< cite "hamilton1971" >}} put forth the theory that selfish incentives actually *lead* animals to engage in social behavior and herding. He posits that each animal seeks to reduce its own "domain of danger", the physical area for which they are the most likely prey. He illustrates this nicely through a 1D example of frogs sitting around a circular pond, where a water snake emerges at a random location and attacks the closest frog. Each frog can achieve its minimal domain of danger by positioning itself in between other frogs, leading to clustering behavior.

{{< figure src="/blog/images/selfish_herd_frogs.png" >}}
Frogs reduce their domain of danger by clustering close to other frogs in Hamilton's selfish herd theory. Adapted from {{< cite "hamilton1971">}}.
{{< /figure >}}

The Boids (bird-oids) simulations {{< cite "reynolds1987" >}} further demonstrated that flocking behavior in birds can be reproduced by specifying 3 simple rules for each agent's behavior: maintaining (1) reasonable separation between itself and its neighbors, (2) alignment with neighbors' heading angles, and (3) cohesion with its neighbors. Most importantly, just as hypothesized for animal groups, no group-level control mechanism is needed to produce flocking behavior -- it is sufficient to simply specify agent-level behavior and rewards.

## Emergence of structure as opposed to chaos depends on the environment

A simple model for collective behavior introduced in 1995 by Vicsek, et al. {{< cite "vicsek1995" >}} uncovers a phase transition from random chaotic movement to the emergence of flocking exclusively through altering environmental parameters. The Vicsek model holds for any self-propelled particles (animals, bacteria, etc.) and assumes simply that, at each timestep, each particle will align its heading angle with its neighbors' headings.

Specifically, each particle $i$ has a position $\xbf_i(t)$ and velocity $\vbf_i(t)$, where the vector $\vbf_i(t)$ is defined by velocity value $v$ (assumed global to all particles) and heading angle $\theta_i(t)$ at time $t$. At each timestep, each particle adjusts its heading angle and position as follows:
\begin{align*}
\theta_i(t+1) &= \langle \theta(t) \rangle_{r,i} + \eta_i(t) \cr
\xbf_i(t+1)   &= \xbf_i(t) + \vbf_i(t) \Delta t
\end{align*}
where $\langle \theta(t) \rangle_{r,i}$ is the average heading of all neighbors $\theta_j(t)$ where particle $j$ is within some radius $r$ of particle $i$, and $\eta_i(t) \sim \textrm{Uni}(-\eta, \eta)$ is a randomly chosen noise/temperature factor with amplitude parameter $\eta$.
$\eta$ can be interpreted as noise in the communication channel between particles, or as the temperature parameter in an Ising-style model.
The key result from Vicsek et al. is that noise values lower than some critical value $\eta_c$ (and choices of particle density $\rho$ greater than critical density $\rho_c$) will induce a phase transition from complete disorder to an ordered state of synchronized collective transport.  

{{< figure src="/blog/images/vicsek.png" >}}
The Vicsek model predicts a phase transition from a disordered state (left) to an ordered state with bulk flow (right) as environmental parameters (density and noise) change. Adapted from {{< cite "vicsek1995">}}.
{{< /figure >}}

Notably, this phase transition is induced by altering *environmental parameters only* ($\eta, \rho$), while the incentives/behavior of the particles remains unchanged. More sophisticated models proposed in later work (e.g., {{< cite "FLIERL1999397" >}}, {{< cite "cucker_smale" >}}) further bolstered these conclusions. In particular, Cucker & Smale (2007) {{< cite "cucker_smale" >}} proved that, under certain communication conditions (interaction strength vs distance), a system of agents will unconditionally converge to a common collective velocity.

---

In this post, we set the scene for understanding emergence in a variety of systems with and without intelligent agents. We saw that:
1) Complexity is an intrinsic property of systems, even simple ones with no agents.
2) Complex emergent behavior can arise from a collection of purely self-interested intelligent agents.
3) The environment (e.g., number density of agents, noise in the system) determines whether structure or chaos emerges.

Designing for emergence and the separation of environmental impact/feedback from individual agent rewards is a major departure from the typical discourse on agentic systems and reinforcement learning, and I look forward to delving into this further in the next post.

Thanks for reading and stay tuned for part 2!

---

Shoutout to my draft readers Alex Wang, Tiger Lu, & Soichiro Hattori!!

---
{{< bibliography src="emergence" >}}
