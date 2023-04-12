### Part 3
1. The empirical analogue would be to replace each expected value with the mean:$$\text{MMD}[\mathcal{F},p,q]:=\sup_{f\in\mathcal{F}}\frac{1}{|X|}\sum_{x\in X}f(x) - \frac{1}{|Y|}\sum_{y\in Y}f(y).$$
2. Note that$$\mathbb{E}_{x\sim p}[f(x)]=\mathbb{E}_{x\sim p}\langle f,\phi(x)\rangle_{\mathcal{H}}=\langle f,\mathbb{E}_{x\sim p}[\phi(x)]\rangle_{\mathcal{H}},$$and similar for $y,q$. So $$\text{MMD}[\mathcal{F},p,q]=\langle f,\mathbb{E}_{x\sim p}[\phi(x)]-\mathbb{E}_{y\sim q}[\phi(y)]\rangle_{\mathcal{H}}.$$Squaring we get $$\begin{align*}\text{MMD}^2[\mathcal{F},p,q]&=\langle f,E\rangle_{\mathcal{H}}^2\\&\le\langle E, E\rangle_{\mathcal{H}}\end{align*},$$where $E=\mathbb{E}_{x\sim p}[\phi(x)]-\mathbb{E}_{y\sim q}[\phi(y)]$ and the inequality follows from Cauchy-Schwarz and $\langle f,f\rangle_\mathcal{H}\le 1.$
3. The empirical analogue is$$\text{MMD}[\mathcal{F},p,q]:=\sup_{f\in \mathcal{F}}\left\langle f, \frac{1}{|X|}\sum_{x\in X}\phi(x)-\frac{1}{|Y|}\sum_{y\in Y}\phi(y)\right\rangle.$$Let $k(x, y) =\langle\phi(x), \phi(y)\rangle$ be our kernel function. From the previous problem, we have the upper bound for $\text{MMD}^2$: $$\le\langle E, E\rangle=\frac{1}{|X|^2}\sum_{x\in X}\sum_{x'\in X}k(x,x')+\frac{1}{|Y|^2}\sum_{y\in Y}\sum_{y'\in Y}k(y,y')-\frac{2}{|X||Y|}\sum_{x\in X}\sum_{y\in Y}k(x,y).$$
4. 