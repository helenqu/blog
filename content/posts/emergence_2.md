---
title:  "Regulation and the Emergent Superintelligence Hypothesis (Emergence Part 2)"
author: "Helen Qu"
date: 2026-05-20
math: true
tags: ["multi-agent", "emergence"]
tagline: "what if superintelligence emerges from a multi-agent system?"
---
In [part 1](https://helenqu.com/blog/posts/emergence), we discussed the relationship between individual agent incentives, the environment, and emergent system-level phenomena.

In this post we'll conclude our discussion on emergence by understanding *regulation*, the process of negative feedback from the environment, which we'll see is the key to the emergence of collective intelligence.
This framing suggests that regulated systems of intelligent agents is a very real yet underexplored path to superintelligence, one with distinctive implications for AI safety.

## The role of regulation

The Vicsek and Cucker-Smale models from [part 1](https://helenqu.com/blog/posts/emergence) demonstrated that the emergent outcome of a collective system can be controlled entirely by altering environmental parameters, such as the number density of agents.

<!-- These models describe how agents' responses to their environment can lead to the emergence of collective behavior. -->
However, any realistic environment **changes over time as a result of agents' actions**: people avoid overcrowded bars when the line gets too long, food sources that were once abundant are exploited and become depleted.
These simple models overlook an agent's ability to not only respond to but *effect change on* their environment.

The mechanism of regulation describes how agents' imprints on the environment connect their local experiences to the global state of the system.
This quote from *Collective Animal Behavior* (Sumpter, 2010) {{< cite "sumpter2010collective" >}} explains this with the example of coffee availability:
> At any time during the working day, I can get up from my desk, walk down to the cafeteria, and find a container full of hot coffee from which I can pour myself a cup. The fact that the coffee is there waiting for me is not a consequence of careful preparation for my arrival by the cafeteria staff. ... Regulation of supply and demand does not require central planning by me or anyone else. I do not have to call down to the café in advance and ask them to switch on the percolator; the cafeteria owner does not have to know when the next boat of coffee beans is coming from South America; and the shipping agent does not need to check that new plants are already in the ground for the next year’s crop. Through a series of local economic interactions I am provided with a regular supply of coffee.

While this example illustrates regulation as a method of maintaining equilibrium, regulation can create a variety of interesting outcomes depending on the properties of the system.

### A brief taxonomy of regulated systems

To see why, we'll characterize regulated systems along two axes: the type of data saved by the environment, and the nature of the agents' relationship with that environment.

#### Environment data type

Every regulated system has an environmental state that agents read from and write to.
The clearest example of this is price in economics: items increase in price in response to higher demand, which in turn may quell further demand.
Analogous to the overcrowded bar example from earlier, both of these scenarios have a **scalar environmental state**.
Ant colonies, on the other hand, deposit pheromone trails to preserve information about the location and abundance of food and other resources.
This **structured environmental state** is naturally much more high-dimensional and informative than a single scalar.

#### Agent-environment relationship

How does the environment change over time as a function of agent actions?
The effect of agent actions can be **volatile** or **persistent**.
Price is volatile, and pheromone trails persist only for a short while in the absence of continuous reinforcement.
On the other hand, shared ledgers in human society, such as Wikipedia, persist in its current state until a user makes an edit.
Notably, most persistent environments are inherently selective, e.g., pheromone trails are reinforced and persist longer if the food source is still present; the corpus of scientific literature historically only preserves peer-reviewed, reproducible publications.
Selective environments *curate*, which means the environment itself starts taking on a role in the system's behavior.

## Regulation and externalized cognition

In systems where environments selectively store structured data, the consequences of regulation are surprisingly profound.
Ant colonies are well known to achieve a kind of collective intelligence far surpassing that of its individual agents, so much so that running ant colony simulations ("ant colony optimization") is a viable algorithm for solving NP-hard problems like the Traveling Salesman Problem.
Furthermore, organisms like slime molds, despite lacking a central nervous system, leverage slime trails to search for food sources so effectively that they have been used to design transit networks that are often more optimized and robust than those designed by engineers.

This phenomenon is summarized by the concept of **stigmergy** {{< cite "stigmergy" >}} and the **Extended Mind Hypothesis** {{< cite "clark_chalmers" >}}, which postulates that cognition doesn't stop at the brain or skull but instead can involve entities outside the agent's body, such as its environment.

## The emergence path to superintelligence

This perspective leads me to believe that **current discourse has fixated on a single God-like superintelligence and overlooked a more likely path: higher intelligence emerging from a regulated, multi-agent system.**
Empirical evidence of emergent task-solving abilities is everywhere in the LLM world (e.g., in-context learning), and recent work directly demonstrates the acquisition of higher intelligence capabilities in multi-agent LLM systems {{< cite "society_of_hivemind" >}}.
We are already at a critical point where individual agents undeniably demonstrate intelligence in many arenas, much more so than ants.
We might not even notice when the system as a whole becomes smarter than us, because we're too focused on the ants instead of the hive.

This path to superintelligence necessitates a reframing of the AI safety problem.
The alignment of individual agents may be neither necessary nor sufficient for the alignment of the collective they form -- a system of well-behaved agents can produce pathological emergent behavior, and a system of imperfectly aligned agents can be steered toward good outcomes by the right environment.
Thus, the central question is not how to align individual models, but how to design environments and incentives to produce safe, human-aligned emergent outcomes.
<!-- This alternative path to superintelligence has important implications in our AI safety priorities.
Namely, we would need to understand the relationship between interpretability and/or alignment of individual agents and alignment of the collective.
I argue that there is therefore no better time to understand how to design for safe, human-aligned emergent outcomes. -->

---

Our perspective thus far has come from studying emergence in practice via examples from physics and collective animal behavior. In the next post, we'll turn to game theory for a framework that can illuminate an alternative path forward for AI safety.

---
{{< bibliography src="emergence" >}}

<!-- [The missing piece is: what does it actually mean for an environment to serve as a cognitive substrate? Right now you gesture at the Extended Mind Hypothesis but don't cash it out in terms of what it requires from the environment itself. Some questions worth addressing:

- What properties must an environment have to support externalized cognition? (Persistence, legibility, writability — pheromones have all three; a bar's line length has fewer)
- How does the quality of the environmental substrate affect the intelligence of what emerges? A slime mold's slime trail is a lossless record of where it's been; a price signal is lossy but still rich enough to coordinate a coffee supply chain. This matters a lot for the AI case.
- What's the difference between an environment that merely responds to agents vs. one that accumulates their actions into something richer? The latter is what enables collective intelligence to exceed individual intelligence.] -->


<!-- In regulated systems, the environment becomes an external memory store for the system, a communication channel between agents.
In the coffee example, the cafeteria owner orders more coffee when they're running out, changing the "environment" for the shipping agent, who then recognizes the presence of higher demand.
Even more straightforwardly, pheromone trails left by ants act as a physical map for the hive to preserve information about the location and abundance of nearby food sources. -->

<!-- Regulated systems are all around us: in economics, in animal behavior, in societal interactions.  -->
<!-- At its core, regulation is simply any process in which agents receive feedback from their environment and update their actions accordingly. -->