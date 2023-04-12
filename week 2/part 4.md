### Part 4
I'm assuming that equality conditions are supposed to be $h_i(x)=0\ \forall i\in [k]$, because otherwise the notation is quite confusing.
1. The Lagrangian is$$\begin{align*}L(x, \lambda)&=f(x) + [g, h]^T\lambda\\&=f(x)+\sum_{i=1}^m\lambda_ig_i(x)+\sum_{i=1}^k\lambda_{i+m}h_i(x).\end{align*}$$
2. As each $g_i \le 0$ we have $$L(x, \lambda)\le f(x)+\sum_{i=1}^k\lambda_{i+m}h_i(x)$$For the optimal $x^*$ in the primal problem, we have $h_i(x^*)=0$, so$$L(x^*,\lambda)\le f(x^*).$$Then$$\bar{L}(\lambda)=\inf_xL(x, \lambda)\le L(x^*,\lambda)\le \inf_xf(x).$$Also,$$\sup_{\lambda_1,\lambda_2,\dots,\lambda_m\ge0}\bar{L}(\lambda)\le\sup_{\lambda_1,\lambda_2,\dots,\lambda_m\ge0}L(x^*,\lambda)\le f(x^*).$$
3. If$$L(x^*,\lambda)\le L(x^*,\lambda^*)\quad\forall \lambda\in\mathbb{R}_+^m\times\mathbb{R}^k,$$then $\sum_{i=1}^m\lambda_i^*g_i(x^*) =0$ or else we could further increase $L(x^*,\lambda^*)$ by decreasing the offending $\lambda_i^*$ (where $g_i(x^*)<0$). If any $h_i(x^*)\ne 0$, then there is no saddle point, as we can set $\lambda_{i+m}$ to $\pm \infty$, so they must all equal $0$. Therefore, $$f(x^*)=L(x^*,\lambda^*)$$
4. The right hand of the saddle point gives $$f(x^*)=L(x^*, \lambda^*)=\bar{L}(\lambda^*),$$but from part 4.2 above we know this is a lower bound on the primal. As it is achievable, it is the optimum solution.
5. The KKT conditions are:
	1. Stationarity: The optimum $x^*$ satisfies $\nabla f+\lambda^T[\nabla g, \nabla h] = 0.$
	2. Primal feasibility: We need $g_i(x^*)\le 0$ and $h_i(x^*)=0.$
	3. Dual feasibility: We need $\lambda_i\ge 0,i\in[m].$
	4. Complementary slackness: We need $\lambda_ig_i(x^*)=0, i\in[m].$
6. From primal feasibility, we have $g_i(x^*)\le 0$ and $h_i(x^*)=0$. So$$L(x^*,\lambda)\le f(x^*)=L(x^*,\lambda^*)$$with equality only when $\lambda_ig_i(x^*)=0,i\in [m]$.
7. We are given that $g, h$ are all convex functions (as affine is convex too). A linear combination of convex functions is convex, so $L$ is convex in $x$. A bounded convex function has exactly one minima, so from dual feasibility (i.e. bounding) there is one minimum for $L(x, \lambda^*)$, which implies the right half of the saddle point condition should be satisfied.