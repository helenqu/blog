---
title:  "An Emergence Perspective on Multi-Agent System Design (Part 2)"
author: "Helen Qu"
date: 2026-03-17
math: true
tags: ["rl", "multi-agent", "emergence"]
draft: true
tagline: "what can emergent behavior in the natural world teach us about multi-agent system design and multi-agent reinforcement learning?"
---
In part 1, we discussed emergence as a property of complex systems and the relationship between individual agent incentives, the environment, and emergent global phenomena. In this post, we will add nuance to this relationship and apply these learnings to the design of multi-agent systems.

## The role of regulation

The models we've discussed so far describe how agents' responses to their environment can lead to the emergence of collective behavior.
However, any realistic environment changes over time as a result of agents' actions: people avoid overcrowded bars when the line gets too long, food sources that were once abundant are exploited and become depleted.
These simple models overlook an agent's ability to not only respond to but *effect change on* their environment.

The mechanism of regulation describes how agents' imprints on the environment connect their local experiences to the global state of the system.
This quote from *Collective Animal Behavior* (Sumpter, 2010) {{< cite "sumpter2010collective" >}} explains this with the example of coffee availability:
> Regulation of supply and demand does not require central planning by me or anyone else. I do not have to call down to the café in advance and ask them to switch on the percolator; the cafeteria owner does not have to know when the next boat of coffee beans is coming from South America; and the shipping agent does not need to check that new plants are already in the ground for the next year’s crop. Through a series of local economic interactions I am provided with a regular supply of coffee.

Regulation changes the story of collective behavior in fascinating and important ways. 

### Passive vs. active regulation

In Section 3, we saw how emergent system-level behavior is affected by changing environmental parameters alone. Now, we'll see the impact to emergent collective behavior of the full feedback cycle: agents' actions change the environment, which in turn affects agent actions.

Let's imagine a swarm of scout honeybees exploiting a flower patch for nectar. Individual bees must decide their next steps only having experienced the availability of nectar in an individual flower.

In *passive* regulation, if two or more bees choose the same flower they simply would not return to that flower, concluding that the resource is overcrowded. *Active* regulation, on the other hand, involves recruitment. In addition to avoiding overcrowded resources, bees assume that an unattended/unexploited flower means excess capacity and will recruit additional bees to the resource.

We can model both of these situations mathematically to approximate the expected behavior. We first assume that the number of individuals choosing a particular resource is Poisson distributed, i.e.,
$$p_k = \frac{(x_t / n)^k}{k!}e^{-x_t / n}$$
In the passive regulation paradigm, we can describe the number of individuals visiting a resource $x_t$ through time $t$ as
$$x_{t+1} = \alpha + p_1 n = \alpha + x_t e^{-x_t / n}$$
where $\alpha$ represents the assumed constant flow of individuals who visit the resource and $n$ is the number of resources (flowers).
In the active regulation story, we assume individuals who encounter a "free" flower will recruit $b-1$ additional individuals to exploit it, resulting in $b$ total individuals at the resource:
$$x_{t+1} = b p_1 n = b x_t e^{-x_t / n}$$

We can see the evolution of the bee population at this flower patch over time in MAKE PLOT 

### Regulation and externalized cognition

In regulated systems, the environment becomes an external memory store for the system, a communication channel between agents.
In the coffee example, the cafeteria owner orders more coffee when they're running out, changing the "environment" for the shipping agent, who then recognizes the presence of higher demand.
Even more straightforwardly, pheromone trails left by ants act as a physical map for the hive to preserve information about the location and abundance of nearby food sources.

- collective behavior <> parallelized sampling/computation

<!-- Even more simply,   -->

<!-- The system's state is determined by the absolute velocity ($v$), the noise level ($\eta$), and the particle density ($\rho$). While $v$ and $r$ define the scale of movement and interaction, $\eta$ and $\rho$ are the critical control parameters that trigger the transition from disordered motion to collective transport. -->

<!-- This model is governed by four parameters: global velocity $v$, interaction radius $r$, temperature factor $\eta$, and density of particles $\rho$. -->