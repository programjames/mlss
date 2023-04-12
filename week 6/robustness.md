## Robustness
1. 'Tis $d\varepsilon$, $\varepsilon\sqrt{d}$, and $\varepsilon$ respectively. The perturbation is larger for $p=1$.
2. Beta(1, 1) is uniform, Beta(5, 5) has a peak at the center, and Beta(0.5, 0.5) has a peak at the corners.
3. (a) False (b) True (c) True (d) True
4. The best $\ell_2$ norm will be minimized when $x$ is the projection of $x_0$ onto the line $w^Tx+w_0 = 0$, and $x=x_0 + \delta w$. I.e. $$w^T(x_0+\delta w)+w_0 = 0\Longleftrightarrow \delta = \frac{-w_0-w^Tx_0}{\lVert w\rVert^2_2}.$$Plugging this in gives $x_{\text{adv}} = x_0-\left(\frac{w^Tx_0+w_0}{\lVert w\rVert_2}\right)\frac{w}{\lVert w\rVert_2},$ or (c).
5. (a) False
   (b) False
   (c) True
   (d) True
6. (a) False, but it's almost good enough as models usually have similar weaknesses even if they have completely different parameters.
   (b) True
   (c) True
   (d) False
7. (a) False, but only because they norm the error after every step of PGD. That's basically the only difference.
   (b) True. E.g. cropping a bunch or cutting out scenes.
   (c) We have $$\nabla_xL = -yw\frac{e^{-yw^Tx}}{1+e^{-yw^Tx}}.$$The fraction is always positive, so the sign is $-y\text{sign}(w)$.
   (d) True
   (e) True
8. (a) False
   (b) False
   (c) False
   (d) False
   (e) True