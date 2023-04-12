# HW 1, Part 1
## By James Camacho
**Part 1.1, Mixture of Bernoullis**
1. The probability we get $x_d$ given $p_d$ is $x_dp_d + (1-x_d)(1-p_d)$, i.e. $p_d$ if $x_d$ is one, otherwise $1-p_d$. Note we can sum (as opposed to using cases) as one of the terms is always zero. We have $P(x|p)$ is the product of all these, or
$$P(x|p) = \prod_{d=1}^{D}x_dp_d + (1-x_d)(1-p_d).$$
2. Basically by definition,
$$\begin{align*}
P(x^{(i)}|\textbf{p},\pi) = \sum_{k=1}^K\pi(k)P(x^{(i)}|p^{(k)})
\end{align*}$$
3. It's the product for each $x^{(i)}\in X$, so taking logs we get
$$\log P(X|\textbf{p},\pi) = \sum_{i=1}^{n}\log P(x^{(i)}|\textbf{p},\pi)$$
**Part 1.2, Expectation Step**
1. Each of the $k$ elements in $z^{(i)}$ are chosen based on the distribution $\pi$. So
$$P(z^{(i)}|\pi) = \prod_{k=1}^Kz_k^{(i)}\pi(k) + \left(1-z_k^{(i)}\right)\left(1-\pi(k)\right).$$
We have
$$P(x^{(i)}|z^{(i)},\textbf{p},\pi)=\sum_{k=1}^K P(x^{(i)}|p^{(k)})z_k^{(i)}$$
(or alternatively $\prod_{k=1}^{K}P(x^{(i)}|p^{(k)})^{z_k^{(i)}}$). Note that $\pi$ doesn't play a role because we are already given which Bernoulli distribution $x$ is drawn from with the indicator vector $z^{(i)}$.
2. Note that
$$P(x^{(i)},z^{(i)}|\textbf{p},\pi) = P(z^{(i)}|\pi)\cdot P(x^{(i)}|z^{(i)},\textbf{p},\pi).$$
So $P(Z,X|\pi,\textbf{p})$ will be the product for each $i$, i.e.
$$\prod_{i=1}^n P(z^{(i)}|\pi)\cdot P(x^{(i)}|z^{(i)},\textbf{p},\pi)$$
3. I'm assuming $\pi_k = \pi(k)$. So, remember how I used $$x_dp_d + (1-x_d)(1-p_d)$$in 1.1.1? Well it's the exact same value as $$p_d^{x_d}(1-p_d)^{1-x_d}.$$Now, $E[z_k^{(i)}|x^{(i)},\pi,\textbf{p}]$ is the probability $p^{(k)}$ was chosen ($\pi_k$) times the probability we get $x^{(i)}$ given $p^{(k)}$ ($P(x^{(i)}|p^{(k)})$, see 1.1.1) divided by the probability we get $x^{(i)}$ (see 1.1.2). This works out to the given expression for $\eta$:
$$\eta(z_k^{(i)})=\frac{\pi_k\prod_{d=1}^{D}(p_d^{(k)})^{(x_d^{(i)})}(1-p_d^{(k)})^{1-x_d^{(i)}}}{\sum_j\pi_j\prod_{d=1}^{D}(p_d^{(j)})^{(x_d^{(i)})}(1-p_d^{(j)})^{1-x_d^{(i)}}}$$ Now,
$$\begin{align*}\log P(Z,X|\tilde{\textbf{p}},\tilde{\pi}) &= \sum_{i=1}^N\log P(z^{(i)}|\tilde{\pi})+\log P(x^{(i)}|z^{(i)},\tilde{\textbf{p}},\tilde{\pi})\\&=\sum_{i=1}^{N}\sum_{k=1}^{K}\left[\log \tilde{\pi}_k + \sum_{d=1}^{D}\left(x_d^{(i)}\log\tilde{p}_d^{(k)} + (1-x_d^{(i)})\log(1-\tilde{p}_d^{(k)})\right)\right]\end{align*}$$To find that big expected value, we have to weight the summation by the chance we actually get the summand in question, or $\eta(z_k^{(i)})$. This gives us the desired:
$$E[\log P(Z,X|\tilde{\textbf{p}},\tilde{\pi})\ |\ X,\textbf{p},\pi] = \sum_{i=1}^{N}\sum_{k=1}^{K}\eta(z_k^{(i)})\left[\log \tilde{\pi}_k + \sum_{d=1}^{D}\left(x_d^{(i)}\log\tilde{p}_d^{(k)} + (1-x_d^{(i)})\log(1-\tilde{p}_d^{(k)})\right)\right]$$

**Part 1.3, Maximization step**

1. Taking a gradient with respect to $\tilde{p}^{(k)}$ gives $$\sum_{i=1}^N\eta(z_k^{(i)})\left[\frac{x^{(i)}}{\tilde{p}^{(k)}} - \frac{(1-x^{(i)})}{1-\tilde{p}^{(k)}}\right]$$ which we want equal to zero. Clearing denominators and a little algebra gives $$\tilde{p}^{(k)} = \frac{\sum_{i=1}^N\eta(z_k^{(i)})x^{(i)}}{\sum_{i=1}^N\eta(z_k^{(i)})}.$$
2. Taking a gradient with respect to $\tilde{\pi}_k$ gives $$\frac{1}{\tilde{\pi}_k}\sum_{i=1}^N\eta(z_k^{(i)}).$$ We have the constraint $\sum_k \tilde{\pi}_k = 1$, and taking a gradient of that gives $1$. So Lagrange multipliers tells us $$\frac{1}{\tilde{\pi}_k}\sum_{i=1}^N\eta(z_k^{(i)}) = \lambda$$ for a constant $\lambda$, which we can solve to be $1/\sum_{k'}N_{k'}$. This gives the desired $$\tilde{\pi}_k = \frac{N_k}{\sum_{k'}N_{k'}}.$$ Note: $N_k = \sum_{i=1}^{N}\eta(z^{(i)}_k)$.  