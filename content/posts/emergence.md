---
title:  "An Emergence Perspective on Multi-Agent System Design"
author: "Helen Qu"
date: 2026-01-06
math: true
tags: ["rl", "multi-agent", "emergence"]
draft: true
tagline: "what can emergent behavior in the natural world teach us about multi-agent system design and multi-agent reinforcement learning?"
---

We observe emergent behavior in stunningly diverse parts of our world, ranging from collective animal behavior to economic systems. Emergence (as defined by Wikipedia) "occurs when a complex entity has properties or behaviors that its parts do not have on their own, and emerge only when they interact in a wider whole". In other words, emergent (or system-level) outcomes can be more than (and different from {{< cite "anderson1972" >}}) the sum of their parts. A wide range of system-level outcomes can emerge from a collection of agents following their own, often simple, reward/value functions. 

On one hand, structure can emerge: for example, migratory birds fly in a V formation despite the lack of system-level organization. On the other hand is the emergence of chaos: nonlinearity and chaotic behavior can emerge from a composition of simple linear functions. **As agentic systems gain prominence, I posit that the emergence of system-level phenomena should be demystified: we should be able to architect systems that sidestep chaotic, negative system-level outcomes while allowing positive outcomes to come "for free".**

We start by seeing that complexity can emerge in the simplest of environments, sometimes with no players at all. We then move on to multi-agent systems through illustrative examples in collective animal behavior, then move on to connections in game theory and mean-field reinforcement learning.

## Self-organization leads to complexity

The canonical example of emergent complexity is Conway's Game of Life, a zero-player game modeled by cellular automata that simply evolves according to fixed rules starting from a user-defined initial state. The game takes place on an infinite 2D grid where each cell is "alive" or "dead" depending on the states of its neighbors. Despite the simplicity of the rules and setup, an incredible taxonomy of "organisms" defined by unique patterns have been shown to emerge in Conway's Game of Life, and is even theoretically proven to be a universal Turing machine. This and other cellular automaton models demonstrate that the emergence of complexity in systems occurs independently of the presence of anything resembling intelligence or incentive structure.

However, even without insidious actors or misaligned incentives, many of these systems tend toward chaos. Certain dynamical systems (e.g., piles of sand) tend toward criticality: sandpiles accumulate sand until a critical point, where the next grain of sand triggers a massive avalanche (modeled by the Abelian/Bak-Tang-Wiesenfeld model {{< cite "bak1987" >}} in the self-organized criticality literature). Thus, we see that bad actors are not the only way systems devolve into chaos: some systems tend toward criticality *by design*.

However, let's consider the alternatives to such a tendency. The behavior of a cellular automaton model can be expressed along a spectrum, where in one extreme all cells either rapidly die or settle into repeating, predictable patterns: the "ordered" extreme; while the other extreme is complete unpredictable randomness, which is equally uninteresting. Somewhere in between is the critical regime: stable enough where information can persist and propagate, but flexible enough to avoid fully predictable/ordered solutions. In fact, "Computation at the Edge of Chaos" {{< cite "langton1990" >}} postulates that a key characteristic of systems with interesting emergent properties *is* existence in this critical regime (at the "edge of chaos"). We will see in future sections how this language of criticality and phase transitions can be a useful ontology for thinking about dynamics in multi-agent systems.

{{< figure src="/blog/images/edge_of_chaos2.png" >}}
The spectrum of behavior in 1D cellular automata, parameterized by $\lambda$. At small values of $\lambda$, cells die out quickly or fall into repeating patterns. Interesting fractal patterns emerge, then dominate, as $\lambda$ increases. However, randomness and chaos begin to take hold as well, and dominate by $\lambda=0.75$. Adapted from {{< cite "langton1990" >}}.
{{< /figure >}}

## Complexity can emerge from selfish intelligent agents

The spontaneous emergence of complex group behaviors in animals is an ideal setting in which to study the relationship between system-level and agent-level behavior. In the absence of explicit leadership, birds self-organize to fly in formation, animals travel in herds, ants form long meandering trails to transport food, and fish swim in schools that move as one to adeptly avoid predation.

We first take the example of herding. In "Geometry of the Selfish Herd" (1971), Hamilton {{< cite "hamilton1971" >}} put forth the theory that animals engage in social behavior and herding due to purely selfish, individual incentives. He posits that each animal seeks to reduce its own "domain of danger", the physical area for which they are the most likely prey. This is illustrated nicely using a 1D example of frogs sitting around a circular pond, where a water snake emerges at a random location and attacks the closest frog. Each frog can achieve its minimal domain of danger by positioning itself in between other frogs, leading to clustering behavior.

{{< figure src="/blog/images/selfish_herd_frogs.png" >}}
Frogs reduce their domain of danger by clustering close to other frogs in Hamilton's selfish herd theory. Adapted from {{< cite "hamilton1971">}}.
{{< /figure >}}

Reynolds {{< cite "reynolds1987" >}} further demonstrated in the Boids (bird-oids) simulations that flocking behavior in birds can be reproduced by specifying 3 simple rules for each agent's behavior: maintaining (1) reasonable separation between itself and its neighbors, (2) alignment with neighbors' headings, and (3) cohesion with its neighbors. Most importantly, just as hypothesized for animal groups, no group-level control mechanism is needed to produce flocking behavior -- it is sufficient to simply specify agent-level behavior and rewards.

## Emergence of structure as opposed to chaos depends on the environment

A simple model for collective behavior introduced in 1995 by Vicsek, et al. {{< cite "vicsek1995" >}} uncovers a phase transition from random chaotic movement to the emergence of flocking exclusively through altering environmental parameters. The Vicsek model is for any self-propelled particles (animals, bacteria, etc.) and assumes simply that, at each timestep, each particle will align its heading with its neighbors' headings.

Specifically, each particle $i$ has a velocity $\vbf_i(t)$, consisting of velocity value $v$ (assumed global to all particles) and heading $\theta_i(t)$ at time $t$. At each timestep, each particle adjusts its heading and position as follows:
\begin{align*}
\theta_i(t+1) &= \langle \theta(t) \rangle_r + \eta_i(t) \cr
\xbf_i(t+1)   &= \xbf_i(t) + \vbf_i(t) \Delta t
\end{align*}
where $\langle \theta(t) \rangle_r$ is the average heading of all neighbors $\theta_j(t)$ in which particle $j$ is within some radius $r$ of particle $i$, and $\eta_i(t) \sim \textrm{Uni}(-\eta, \eta)$ is a randomly chosen noise/temperature factor with amplitude parameter $\eta$. 
This model is governed by three parameters: global velocity $v$, interaction radius $r$, and temperature factor $\eta$.

---
acknowledgments

---
{{< bibliography src="emergence" >}}
