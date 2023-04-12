
### Part 2
**2.1, Lemma from Class**

We want to find $\mu = E[y^*|y].$ We have $$E[\mu y^T]=E[E[y^*|y]y^T] = k(X^*,X),$$and $$E[yy^T]=k(X, X).$$So $$\mu = \mu y^T(yy^T)^{-1}y = E[\mu y^T]E[(yy^T)^{-1}]E[y] = k(X^*,X)k(X,X)^{-1}y.$$ We also want to find $\Sigma = E[(y^*-\mu)(y^*-\mu)^T|y].$ I've spent several days on this and haven't got a clue (well, I could use the pdf of the posterior distribution, but that would take forever to write out). I'll just take the loss on these points and look up the solution online.