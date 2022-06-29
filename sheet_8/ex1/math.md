# FTCS

FTCS is a first order method, to solve a pde with a single time derivative. It is a forward in time integration. It basically solves the time integration with a forward finite difference method.
$
\frac{\partial u}{\partial t} = \frac{u_j^{n+1} - u_j^{n}}{\Delta t} + \mathcal{O}(\Delta t)
$
If we know now the rhs of the partial differential equation:

$
\frac{\partial u}{\partial t} = \mathcal{L}\left(u, x, t, \frac{\partial u}{\partial t}, \frac{\partial u ^2}{\partial t ^2}, \dots \right)
$

We can then write:
$
u_j^{n+1} = u_j^{n} + \Delta t \mathcal{L}\left(u, x, t, \frac{\partial u}{\partial x}, \frac{\partial u ^2}{\partial x ^2}, \dots \right)
$

In practice of the advection equation

$
\frac{\partial u(x, t)}{\partial t} + \frac{\partial u(x, t)}{\partial x} = 0 
$

We can then approximate the spacial derivative with a finite difference method $\mathcal{O}(\Delta x^3)$. And calculate it with:

$
u_j^{i+1} = u_j^i + \frac{\Delta t}{2\Delta x} \left( u_{j+1}^i - u_{j-1}^i \right)
$

Always remember, that a second order time derivative can be implemented as two coupled first order time derivatives systems.

$
\begin{align}
\frac{\partial u^2}{\partial t^2} &= \frac{\partial u^2}{\partial x^2} \\
\Longrightarrow & \\
\frac{\partial s}{\partial t} &= \frac{\partial r}{\partial x} \\
\frac{\partial u}{\partial x} &= r \Rightarrow r = \frac{\partial u}{\partial x} \Rightarrow \frac{\partial r}{\partial t} = \frac{\partial }{\partial t}\frac{\partial u}{\partial x} = \frac{\partial }{\partial x}\frac{\partial u}{\partial t} = \frac{\partial s}{\partial x} \\
\frac{\partial u}{\partial t} &= s \\
\end{align}
$

One can easily see, that a second initial condition is missing. we just use $\frac{\partial u}{\partial t} = 0$ for convenience.
