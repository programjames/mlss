# Week 2
## By James Camacho
### Part 1
**1.1, Quantile Regression**
1. Note that$$\frac{\text{d}}{\text{d}x}\rho_\tau(y_i-x) = \begin{cases}-\tau&w<y_i\\1-\tau&w>y_i.\end{cases}$$Let $$f(x) = \sum_{i}\rho_\tau(y_i-x),$$which is differentiable everywhere except when $x=y_i$ for some $y_i$, and has derivative$$\frac{\text{d}}{\text{d}x}f(x) = \sum_{i}I(w > y_i)-N\tau.$$This derivative is positive when $x > y_{\tau}$ and negative when $x<y_{\tau}$ so the minimum of $f(x)$ occurs when $x = y_\tau$.
2. It's equivalent to the one-norm or absolute value, just halved. It will find the median of the data.
3. If we set$$u_i=\begin{cases}y_i-x_i^T\beta&x_i^T\beta \le y_i\\0&x_i^T\beta>y_i,\end{cases}$$and$$v_i=\begin{cases}0&x_i^T\beta \le y_i\\x_i^T\beta-y_i&x_i^T\beta>y_i,\end{cases}$$then$$\sum_{i=1}^N\rho_\tau(y_i-x_i^T\beta)=u^T1\tau+v^T1(1-\tau),$$ $u,v\ge 0,$ and$$X^T\beta-y+u-v=0.$$So$$\arg\min_{\beta\in\mathbb{R}^K}\sum_{i=1}^N\rho_\tau(y_i-x_i^T\beta)\ge \arg\min_{\beta,u,v}u^T1\tau+v^T1(1-\tau)$$given $X^T\beta-y+u-v=0; u,v\ge 0.$ Also any $\beta$ from the RHS can be plugged into the LHS, so the reverse inequality is true as well. The two problems are equivalent.
4. We want to minimax the Lagrangian:$$\max_{a,b,\lambda}\min_{u,v,\beta}u^T1\tau+v^T1(1-\tau)-\lambda^T(X^T\beta-y+u-v)-a^Tu-b^Tv$$where $a, b\ge 0$. Taking a gradient w.r.t. $\beta$ gives$$\lambda^T X^T = 0.$$Taking gradients w.r.t. $u,v$ give$$a+\lambda = 1\tau,$$$$b-\lambda=1-1\tau.$$Plugging this back in gives the maximization problem$$\max_{\lambda}\lambda^Ty.$$subject to $\lambda^TX^T = 0.$ If we let $z=1-1\tau+\lambda$ we get the equivalent problem$$\max_{z}y^Tz,\quad\text{subject to}\ Xz=(1-\tau)X1.$$Note that $z=1-1\tau+\lambda=1-a=b,$ so $0\le z\le 1$ or $z\in[0,1]^n$.
5. From complementary slackness, when $z_i=0$ we have $$a_i=1\implies u_i=0\implies y_i>x_i^T\beta.$$Similarly, when $z_i=1$ we find$$b_i=1\implies v_i=0\implies y_i\le x_i^T\beta.$$ When $z_i\in (0, 1)$ we get both $$a_i, b_i>0\implies u_i=v_i=0\implies y_i=x_i^T\beta.$$
6. See code.

