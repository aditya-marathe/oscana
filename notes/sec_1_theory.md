<!-- markdownlint-disable MD026 -->

# 1. Neutrino Theory

## 1.1 Dirac's Equation

The present Standard Model, is a gauge theory of group $\text{SU}(3) \times \text{SU}(2)_L \times \text{U}(1)_Y$.

The Dirac Lagrangian describes spin-half particles (fermions) which are excitations of the Dirac field $\psi(x)$.

$$
    \mathscr{L} = \bar{\psi}(x) \left( i \gamma^\mu \partial_\mu - m \right) \psi(x) + \mathscr{L}_\text{int}
$$

For now, we can consider non-interacting fermions such that $\mathscr{L}_\text{int} = 0$.

From the Lagrangian, we can infer the energy dimension of $\psi(x)$ and $\bar{\psi}(x)$ is $\frac{3}{2}$, since $[\mathscr{L}] = 4$, $[\partial_\mu] = [m] = 1$ and the $\gamma$ matrices are dimensionless.

The following equation of motion is obtained by treating $\psi(x)$ and $\bar{\psi}(x)$ as independent.

$$
    (i \gamma^\mu \partial_\mu - m) \psi(x) = 0
$$

A Dirac field describing massless fermions and antifermions, the following equation of motion is obtained.

$$
    \gamma^\mu \partial_\mu \psi(x) = 0
$$

A massless Dirac field has chiral symmetry and $\psi(x)$ can be decomposed into the right-handed and left-handed components $\psi_L(x)$ and $\psi_R(x)$, respectively.

$$
    \psi(x) = \psi_L(x) + \psi_R(x) = \left( \hat{P}_L + \hat{P}_R \right) \psi(x)
$$

Here, $P_L = \frac{1}{2}(1 - \gamma^5)$ and $P_R = \frac{1}{2}(1 + \gamma^5)$ are projection operators for the left- and right-handed chiral components.

Chiral symmetry implies that the Lagrangian is invariant under individual rotations of the two chiral components: $ \psi_L(x) \rightarrow \exp(i\theta_L) \psi_L(x), \;\; \psi_R(x) \rightarrow \psi_R(x) $, and $ \psi_L(x) \rightarrow \psi_L(x), \;\; \psi_R(x) \rightarrow \exp(i\theta_R) \psi_R(x)$. This symmetry is demonstrated below.

$$
    \begin{align*}
        \mathscr{L} &= \bar{\psi} \left( i \gamma^\mu \partial_\mu - m \right) \psi \\
        &= \psi^\dag \gamma^0 \left( i \gamma^\mu \partial_\mu - m \right) \psi \\
        &= [\psi_L^\dag + \psi_R^\dag] \gamma^0 \left( i \gamma^\mu \partial_\mu - m \right) [\psi_L + \psi_R] \\
        &= i \left\{ \begin{align*} & \psi_L^\dag \gamma^0 \gamma^\mu \partial_\mu \psi_L \\ &+ \psi_L^\dag \gamma^0 \gamma^\mu \partial_\mu \psi_R \\ &+ \psi_R^\dag \gamma^0 \gamma^\mu \partial_\mu \psi_L \\ &+ \psi_R^\dag \gamma^0 \gamma^\mu \partial_\mu \psi_R \end{align*} \right\} - m \left\{ \begin{align*} & \psi_L^\dag \gamma^0 \psi_L \\ &+ \psi_L^\dag \gamma^0 \psi_R \\ &+ \psi_R^\dag \gamma^0 \psi_L \\ &+ \psi_R^\dag \gamma^0 \psi_R \end{align*} \right\} \\
        &= i \left\{\psi^\dag P_L \gamma^0 \gamma^\mu \partial_\mu P_L \psi + \ldots \right\} - m \left\{ \psi^\dag P_L \gamma^0 P_L \psi + \ldots \right\} \\
        &= i \left\{ \bar{\psi}_L \gamma^\mu \partial_\mu \psi_L + \bar{\psi}_R \gamma^\mu \partial_\mu \psi_R \right\} - m \left\{ \bar{\psi}_L \psi_R + \bar{\psi}_R \psi_L \right\}
    \end{align*}
$$

Above, we have used the property of the projectors that $P_L P_R = P_R P_L = 0$ to simplify. Let us consider, for brevity, only the rotation on the left-handed component:

$$
    \psi_L(x) \rightarrow \psi_L'(x) = \exp(i\theta_L) \psi_L(x), \;\; \psi_R(x) \rightarrow \psi_R'(x) = \psi_R(x)
$$

If massless ($m = 0$), we get the following.

$$
    \begin{align*}
        \mathscr{L}' &= i \bar{\psi}_L' \gamma^\mu \partial_\mu \psi_L' + i \bar{\psi}_R' \gamma^\mu \partial_\mu \psi_R' \\
        &= i \bar{\psi}_L \exp(-i\theta_L) \gamma^\mu \partial_\mu \exp(i\theta_L) \psi_L + i \bar{\psi}_R \gamma^\mu \partial_\mu \psi_R \\
        &= i \bar{\psi}_L \gamma^\mu \partial_\mu \psi_L + i \bar{\psi}_R \gamma^\mu \partial_\mu \psi_R \\
        &= \mathscr{L}
    \end{align*}
$$

Hence, it **is** invariant under the transformation. We can show that the Lagrangian would **not** be invariant for massive fermions ($m \neq 0$).

$$
    \begin{align*}
        \mathscr{L}' &= i \bar{\psi}_L' \gamma^\mu \partial_\mu \psi_L' + i \bar{\psi}_R' \gamma^\mu \partial_\mu \psi_R' - m \left\{ \bar{\psi}_L' \psi_R' + \bar{\psi}_R' \psi_L' \right\} \\
        &= i \bar{\psi}_L \exp(-i\theta_L) \gamma^\mu \partial_\mu \exp(i\theta_L) \psi_L + i \bar{\psi}_R \gamma^\mu \partial_\mu \psi_R \\
        &\;\;\;\;\;- m \left\{ \bar{\psi}_L \exp(-i\theta_L) \psi_R + \bar{\psi}_R \exp(i\theta_L) \psi_L \right\} \\
        &= i \bar{\psi}_L \gamma^\mu \partial_\mu \psi_L + i \bar{\psi}_R \gamma^\mu \partial_\mu \psi_R - m \left\{ \bar{\psi}_L \exp(-i\theta_L) \psi_R + \bar{\psi}_R \exp(i\theta_L) \psi_L \right\} \\
        &\neq \mathscr{L}
    \end{align*}
$$

The same can be shown for the right-handed rotation. Hence, chiral symmetry is broken for massive fermions.

## 1.2 Higgs Mechanism

## 1.3 Lepton Masses
