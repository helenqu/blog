---
title:  "Mechanism Design for AI Safety (Emergence Part 3)"
author: "Helen Qu"
date: 2026-06-04
math: true
tags: ["ai-safety", "multi-agent", "emergence"]
tagline: "how can we design for emergent aligned outcomes?"
---

In previous posts, we've seen how [complex emergent behavior can arise in multi-agent systems](https://helenqu.com/blog/posts/emergence/), and how [distributed cognition can emerge from environmental feedback](https://helenqu.com/blog/posts/emergence_2/).
This led me to [conclude](https://helenqu.com/blog/posts/emergence_2/) that superintelligence is arguably more likely to emerge from a multi-agent system as opposed to in a single God-like superintelligent model, what I called the **emergent superintelligence hypothesis**.
Among other things, this has important implications for the AI safety landscape, namely placing more emphasis on designing for alignment of the *collective* as opposed to individual agents. 

In this post, we'll discuss the motivations for this perspective on AI safety, and how the field of mechanism design can provide a path forward.

## Individual agent alignment is not sufficient

The vast majority of work in AI safety thus far has focused on individually training aligned agents and developing interpretability methods to understand and prevent subversive behavior.
However, we will soon deploy multi-agent systems in increasingly unconstrained environments, giving these systems more freedom and open-ended tasks.
So, regardless of how convincing the emergent superintelligence hypothesis is, I argue that the alignment of the *collective* cannot be ignored.
The most immediate question this raises is: in the hypothetical world in which individual agent alignment is solved, is a collective of perfectly aligned agents guaranteed to be aligned?

I argue that **aligning individual agents is not sufficient (and perhaps not even necessary) to ensure that the multi-agent system is well-aligned with human values**. 

Consider the following example adapted from Conitzer & Oesterheld's 2024 paper {{< cite "Conitzer_Oesterheld_2024" >}}: two agents collaborate to provide a service, each choosing a level of quality $q_i$ with which to perform their part of the service. 
The overall quality of the service $q$ is defined as the lower of the $q_i$, i.e., $q \equiv \text{min}_i \; q_i$, and humans value receiving a high quality service (maximizing $q$). 
Agents are compensated commensurate to the overall service quality $q$ with an additional small bonus $\epsilon$ if they choose a lower quality than their collaborator, resulting in a slight preference toward providing the lower quality service.
Unfortunately, the Nash equilibrium for this game (known as the Travelers' Dilemma) is for both agents to provide the lowest possible service quality ($q_1 = q_2 = 0$).
Thus, while each agent's incentive is largely aligned with human incentives (providing a high overall service quality $q$), the equilibrium game outcome is actually anti-aligned.

Finally, it's important to recognize that different types of alignment appear in multi-agent systems as opposed to individual models. I appreciate the framework introduced by Carichon et al. {{< cite "carichon2025coming" >}}, which lays out 3 different types of alignment:
- **objective:** agents need to cooperate to achieve the objective/goal they were deployed for
- **human:** agents need to achieve the objective without compromising on human values
- **preferential:** the specific user deploying the agents have some specific preferences that the agents' behavior should respect

This framework makes clear that a holistic picture of alignment involves work in _cooperative AI_: designing agents, environments, and institutions that enable cooperation even in complex, mixed-motive settings ({{< cite "Conitzer_Oesterheld_2024" >}}, {{< cite "dafoe2020open" >}}, {{< cite "dafoe2021cooperative" >}}, {{< cite "hammond2025multi" >}}).

<!-- - yes, partially bc there's likely no single vector that defines "aligned with human interests" (each agent is likely to have slighly different incentives than the others, just like humans). consider the travelers problem, where the two agents are each mostly incentivized to care about the overall quality of the service but slightly biased toward themselves providing the lower quality of the two (from conitzer/oesterheld)
- relatedly, cooperative AI deals with all the ways to get agents to cooperate in scenarios where it's mutually beneficial but not straightforward (eg prisoners dilemma, tragedy of the commons/public goods games)
- relationship bt cooperative AI and collective alignment: eg is it required to cooperate to achieve collective alignment? agents could also cooperate to advance a subversive goal -->

## Individual agent alignment may not be necessary

In most human interactions, we can't be certain about our counterparts' true intentions.
Society operates on a system of trust and contracts that align different parties' incentives to make the desired outcome also the rational outcome.
I argue that AI alignment can follow the same principles: even if we aren't fully privy to each model's thought processes and intentions, we can design the system they live in to incentivize human-aligned outcomes.

### The mechanism design perspective

The field of **mechanism design**, sometimes called reverse game theory, offers a path toward such an approach.
As the name suggests, mechanism design starts with a desired outcome and works backwards to design the game, or mechanism, that produces it.

We can illustrate this with a simple, real-world example of buying a car from a used car salesman.
We would like to know the true quality of the used car, but we know the salesman is incentivized to distort the truth.
Though we can't rewire the salesman to tell the truth or hook them up to a polygraph machine, mechanism design shows us that there's another way: we can redesign the game such that the salesman is incentivized to tell the truth.
One possible strategy is to give the salesman two options: either accept a lump sum payment of $x$; or a higher offer $y > x$ but only a small deposit $\epsilon$ is paid initially, and the majority $y-\epsilon$ is paid at a later time when the car has proven reliably functional.
This way, the salesman is incentivized to tell the truth by picking the option that aligns with their car's true quality, potentially accepting a lower cost for a less reliable car.

Notably, **mechanism design is agnostic to players' subversive incentives**.
The goal is simply to create an environment in which any rational player would take the set of actions desired by the game designer.
These are called *incentive compatible* mechanisms, mechanisms in which every player is incentivized to tell the truth rather than any possible alternative.
In other words, we want to design a game in which deception and defection from alignment is always a losing strategy regardless of agents' internal goals.
This framing relaxes the burden currently carried solely by individual model safety efforts, since a system designed this way behaves well even without full transparency into individual agents' incentives.

## Ingredients for a mechanism design approach

<!-- The simplest type of incentive compatible mechanism is the Vickrey-Clarke-Groves mechanism, which designs a payment structure that aligns every player's incentives with the global social welfare. [what does this have to do with telling the truth]

mechanism design is already a theory of designed emergence — it's the study of what local incentive structures reliably produce desired collective outcomes, without any agent needing to represent or intend that collective outcome. That closes the loop from your opening (slime molds, ants) back to your conclusion: just as evolution "designed" pheromone gradients that produce colony-level intelligence, we could in principle design interaction protocols for AI agents that produce system-level alignment, without solving alignment at the individual agent level at all. -->

<!-- Investigating the building blocks of a mechanism design (MD) approach to multi-agent system alignment will help clarify the landscape and identify open questions. -->

### Classical setting

From {{< cite nisan2007algorithmic >}}, classical mechanism design (MD) assumes the following setting:
- There are $n$ players in the game.
- The game involves a set of alternatives $A$ that each player has a private set of preferences on.
- These preferences are encoded in terms of a _valuation function_ $v_i: A \rightarrow \mathbb{R}$ for each player, where $v_i(a), \; a \in A$ denotes the "value" that player $i$ assigns to alternative $a$ being chosen.

The goal is to design a _mechanism_, a (social choice) function that aggregates player preferences and gives them the appropriate incentives to ensure the desired outcome.

### MD for alignment

How can we tailor the classical MD framework to the challenges unique to the AI safety/alignment setting?

I'll offer my perspective of the setting:
- There are $n$ agents in the game. Note that the human-agent 1-1 interaction case is included by setting $n=1$.
- There is a single (or group of) human principal(s), whose objective can be described by a welfare function that the principal wants to maximize. Unlike the classical MD setting, the welfare function is not determined by players' preferences.

We wish to design a mechanism that optimizes this welfare function in a way that is robust to agents' unknown and possibly adversarial preferences. This aligns well with *principal-agent theory* (e.g., {{< cite holstrom1979moral >}}), which centers around a principal-designated objective rather than player preference aggregation; and *robust mechanism design* {{< cite bergemann2005robust >}}, which endeavors to design mechanisms that work across many sets of possible player preferences.

I'll dedicate the next section to what I believe are the most pressing open questions to think about in these early days of the field.

<!-- I contend that almost all of these areas require further work and understanding to tailor to the setting of AI alignment, and outline my thoughts below. -->

### Open questions

#### Classical MD is formulated with respect to preferences over a set of alternatives, but not every interaction between agents (or between agents and humans) can be modeled as picking between alternatives.

Historically, MD applications have been most successful in scenarios where players' preferences are defined with respect to a discrete set of alternatives, e.g., auctions, voting, matching markets, and public goods problems.
The space of natural language interactions is explicitly not this, leading me to believe that safety at the interaction level is still best handled by traditional AI safety methods like RLHF.
MD, however, would excel at orchestrating higher-level agent-agent interactions, such as designing rules for resource allocation amongst agents, collective decision-making, and overall incentive structure.
Of course, this stratification of concerns is imperfect since individual interactions are the building blocks of higher-level/emergent trends, but separating different levels of abstraction is very common in e.g., economic and physical modeling.

#### To what degree are AI agents rational?

An important underlying assumption of game theory and economic modeling is that of *rationality*: rational players will take the course of action that maximizes their utility.
Mechanism design capitalizes on this assumption by aligning the system-level desired outcome with maximum utility actions, thereby assuring the desired outcome when all players act rationally.
However, I believe this could break down in at least two different ways when the players are AI agents:

1) MD rests on the assumption that there is something intrinsically motivating to people (e.g., money). Is there anything we can be sure is “intrinsically motivating” to AI agents?
2) Even if we can identify something intrinsically motivating, or design agents in this way, can we be certain that they'll always act rationally to maximize utility? Relatedly, even humans are known not to act purely rationally, leading to the introduction of *bounded rationality* to explain how most real-world situations involve limitations that prevent humans from making the rationally optimal choice.

  <!-- - Touched on by [https://arxiv.org/pdf/2402.12907](https://arxiv.org/pdf/2402.12907)
  - Related to how the assumption of rationality/bounded rationality may or may not hold (either bc agents are bad or bc we can’t reliably know what they are intrinsically motivated by) -->

#### Selecting the right social choice function is integral to MD (e.g., maximizing the social welfare), but designing an SCF for alignment with human values is nontrivial.

Specifying a set of outcomes aligned with human values is possibly the most vexing philosophical question of AI alignment, and unfortunately MD does not give us an opportunity to sidestep this.
The usual questions arise: Which humans' preferences matter? Can these preferences be encoded in as simple and low-dimensional a framework as a social choice function?

#### Existing work/further reading

In addition to the cooperative AI references above, I wanted to highlight a few papers applying economic principles to AI safety that helped shape my thinking:
- **Distributional AGI Safety** (Tomasev et al., 2025) {{< cite tomavsev2025distributional >}}: Proposes a concrete market design-inspired "defense-in-depth" approach for AGI safety in the scenario where AGI emerges through a patchwork of sub-AGI individual agents (similar to the emergent superintelligence hypothesis).
- **Incomplete Contracting and AI Alignment** (Hadfield-Menell & Hadfield, 2019) {{< cite hadfield2019incomplete >}}: Explicitly concludes that reward misspecification (e.g., the problem of choosing an appropriate social choice function) is an unavoidable reality and study this in the context of incomplete contract theory. Believe that AIs must be built in a way that reads/respects a shared implied societal context the way that humans do.
- **AI Alignment via Incentives and Correction** (Agarwal et al., 2026) {{< cite agarwal2026ai >}}: Introduces an adversarial solver-auditor system and associated RL framework that self-regulates toward alignment through reward shaping.
- **Roadmap on Incentive Compatibility for AI Alignment and Governance in Sociotechnical Systems** (Zhang et al., 2025) {{< cite zhang2024roadmap >}}: Encourages future research in incentive compatibility/mechanism design for AI alignment.
- **Principal-Agent Reinforcement Learning** (Ivanov et al., 2024) {{< cite ivanov2024principal >}}: Concretely extends principal-agent and contract theory to Markov decision processes and reinforcement learning.

## Conclusion

Multi-agent systems will soon become the primary paradigm of human interaction with AI systems, and I argue that this comes with a great variety of possible emergent phenomena that can't be anticipated from studying and developing agents in isolation.
I dedicated this [blog post series](https://helenqu.com/blog/tags/emergence/) to demystifying the process of emergence with the goal of learning how to design for emergent system-level outcomes.
In this post, I argue that alignment will be an emergent property of a multi-agent system and not a foregone conclusion even if all constituent agents are individually aligned. 
I laid out a systems-first perspective on AI alignment centered on concepts borrowed from mechanism design, game theory, and economic modeling.
While there are certainly more questions than answers at this point, I believe that this is an exciting new direction for future thought and research.

<!-- ### More FAQ style
- What about the single superintelligent model? Does mechanism design approach necessitate the emergence superintelligence hypothesis to hold true
  - Principal agent theory (application paper: arXiv:2407.18074) -->

---


If you found this post useful, please cite!

```
@article{qu2026mechanism,
  title   = "Mechanism Design for AI Safety (Emergence Part 3)",
  author  = "Qu, Helen",
  journal = "helenqu.com",
  year    = "2026",
  month   = "June",
  url     = "https://helenqu.com/blog/emergence_3"
}
```
---
{{< bibliography src="emergence" >}}