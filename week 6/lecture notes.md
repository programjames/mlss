## By James Camacho
### Risk Decomposition
- Hazard = danger. Could be bad actors or unknown scenarios.
- Vulnerability = how much harm does hazard cause. Robustness makes ML not vulnerable.
- Exposure = how often will be put in hazardous situation.
- Risk = vulnerabilitiy x exposure x hazard, any item can be decreased to lower risk.

### Accident Models
- Swiss cheese - hazards can get through a small portion of each cheese. If enough protections are stacked, it will hopefully block all hazards.
- Bow tie - Lots of preventative protections (proactive) and protective barriers (reactive).
- Causation isn't very linear or modelled as separate systems working together
- Emergence of properties once enough variables in complex systems. DL models are complex systems.
### Black Swans
- "Things that have never happened before happen all the time" - Scott D. Sagan
- Black swans are outside expectations but could have severe impact (e.g. snowstorm for self-driving cars)
- long-tailed distributions are often max-sum equivalent (e.g. power-law x^(-a))
- long-tailed shows up everywhere in the world (e.g. money a company makes)
- Black swans are "unknown unknowns" and long-tailed events
- AI impact may be long-tailed
### Adversarial Robustness
- Models can be made robust to imperceptible distortions (e.g. cat -> guacamole with tiny change in pixel values), but yet to be made robust to perceptible changes.
- In future, AI may aid in training other AI (e.g. AI could be proxy for human morals). Needs to be robust
- Adversary's goal: given "budget" of distortion, maximize loss
- Protection: Do this and add adversarial examples to training set. Unfortunately can reduce accuracy by 10%.
- Adversarial attacks can often transfer to different models, so can't assume you have black box.
- CutMix is one of the most effective methods for robustness, smooth gradients help too.
- Robustness guarantee - can mathematically prove it will be robust within some area given the weights.
### Black Swan Robustness
- Many models have the same errors even with different architectures.
- Can increase robustness by testing on images lots of models get wrong.
- ANLI - natural language inference dataset to trick NLP models
- Mixup = like cutmix but weighted average of entire pictures
- AutoAugment = trying lots of augmentations to see what works the best, extremely computationally expensive
- AugMix = random augmentations, but still recognizable
- PixMix = mixes training with some other random dataset (e.g. fractals)