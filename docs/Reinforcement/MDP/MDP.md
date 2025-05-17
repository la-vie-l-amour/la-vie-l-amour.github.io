
1.  Probability Theory
    * Conditional independence ($X \bot A | B$)
        $$
        p(X = x|A = a, B = b) = p(X = x|B = b)
        $$
        The memoryless property of the Markov process is as follows:
        $$
        p(s_{t+2}|s_{t+1},s_{t}) = p(s_{t+2}|s_{t+1}) \quad (1)
        $$
        if $s_{t+1}$ was given, then $s_{t+2}$ is conditionally independent of $s_t$.
    * Chain rule of conditional probability and joint probability
        $$
        p(x|a) = \sum_{b}p(x,b|a) = \sum_{b} p(x|b,a)p(b|a) \quad (2)
        $$

    * Conditional expectation
        $$
        \mathbb{E}\left[X|A = a\right] = \sum_{x}xp(x|a) \quad (3)
        $$
        $$
        \mathbb{E}\left[X\right] = \sum_{a} \mathbb{E}\left[X|A = a\right]p(a) \quad (4)
        $$
        $$
        \mathbb{E}\left[X|A = a\right] = \sum_{b}\mathbb{E}\left[X|A = a,B = b\right]p(b|a) \quad (5)
        $$
        The proof of equation (4) is as follows:
        $$
        \begin{align*}
            \sum_{a}\mathbb{E}\left[X|A=a\right]p(a)
            &= \sum_{a}\left[\sum_{x}p(x|a)x\right]p(a) \\
            &= \sum_{a}\sum_{x}p(x|a)p(a)x \\
            &=\sum_{x}\left[\sum_{a}p(x|a)p(a)\right]x \\
            &=\sum_{x}p(x)x \\
            &=\mathbb{E}\left[X\right] \\
        \end{align*}
        $$
        The proof of equation (5) is as follows. It uses equation (2).
        $$
        \begin{align*}
            \sum_{b}\mathbb{E}\left[X|A = a,B = b\right]p(b|a)
            &= \sum_{b}\left[\sum_{x}p(x|a,b)x\right]p(b|a)\\
            &= \sum_{x}\left[\sum_{b}p(x|a,b)p(b|a)\right]x \\
            &= \sum_{x}p(x|a)x \\
            &= \mathbb{E}\left[X|A = a\right]
        \end{align*}
        $$
2.  Bellman Equation
$$
    \begin{align*}
        G_{t}
        &= R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3}+\cdots \\
        &= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots) \\
        &= R_{t+1} + \gamma G_{t+1}
    \end{align*}
    $$
    $G_{t}$ is called return. $R_{t+1}$ is called reward. $\gamma$ is a discount rate.
    $$
    \upsilon_{\pi}(s)\doteq\mathbb{E}\left[G_{t}|S_{t} = s\right]
    $$
    $\upsilon_{\pi}(s)$ is called the state-value function. The state-value function can be written as:
    $$
    \begin{align*}
        \upsilon_{\pi}(s )
        &= \mathbb{E}\left[G_{t}|S_{t} = s\right] \\
        &= \mathbb{E}\left[R_{t+1} + \gamma G_{t+1} | S_{t} = s\right] \\
        &= \underbrace{\mathbb{E}\left[R_{t+1}|S_{t} = s\right]}_{\textcircled{1}} + \gamma\underbrace{ \mathbb{E}\left[G_{t+1}|S_{t} = s\right]}_{\textcircled{2}}
    \end{align*}
    $$
    \textcircled{1} can be written as follows:
    $$
    \begin{align*}
        \mathbb{E}\left[R_{t+1}|S_{t} = s\right]
        &= \sum_{a}\pi(a|s)\mathbb{E}\left[R_{t+1}|S_{t} = s,A_{t} = a\right]  \\
        &= \sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r
    \end{align*}
    $$
    It first uses equation (5), then uses equation (3).

    \textcircled{2} can be written as follows:
    $$
    \begin{align*}
        \mathbb{E}\left[G_{t+1}|S_{t} = s\right]
        &= \sum_{s^{'}}\mathbb{E}\left[G_{t+1}|S_{t} = s,S_{t+1} = s^{'}\right]p(s^{'}|s)\\
        &=\sum_{s^{'}}\mathbb{E}\left[G_{t+1}|S_{t+1} = s^{'}\right]p(s^{'}|s) \\
        &=\sum_{s^{'}}\upsilon_{\pi}(s^{'})\sum_{a}p(s^{'},a|s)\\
        &=\sum_{s^{'}}\upsilon_{\pi}(s^{'})\sum_{a}p(s^{'}|a,s)\pi(a|s)
    \end{align*}
    $$
    It first uses equation (5), secondly, it uses equation (1), then it uses the *Law of Total Probability*, finally, it uses equation (2).
    $$
    \begin{align}
        \upsilon_{\pi}(s)
        &= \textcircled{1} + \gamma\textcircled{2} \nonumber\\
        &= \sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r  + \gamma \sum_{s^{'}}\upsilon_{\pi}(s^{'})\sum_{a}p(s^{'}|a,s)\pi(a|s)\nonumber \\
        &= \sum_{a}\pi(a|s)\left[\sum_{r}p(r|s,a)r + \gamma\sum_{s^{'}}p(s^{'}|a,s)\upsilon_{\pi}(s^{'})\right] \quad (6)
    \end{align}
    $$
    Equation (6) is the Bellman Equation. We show other expressions of the Bellman Equation as follows.

    First, it follows from the *Law of Total Probability* that:
    $$
    \begin{align*}
        p(s^{'}|s,a) &= \sum_{r}p(s^{'},r|s,a) \\
        p(r|s,a) &= \sum_{s^{'}}p(s^{'},r|s,a)
    \end{align*}
    $$
    Then, equation (6) can be rewritten as:
    $$
    \upsilon_{\pi}(s) = \sum_{a}\pi(a|s)\sum_{r}\sum_{s^{'}}p(s^{'},r|s,a)\left[r+\gamma\upsilon_{\pi}(s^{'})\right]
    $$
    The Bellman Equation can be expressed in terms of action values. As we know:
    $$
    \begin{align*}
        q_{\pi}(s,a) &= \mathbb{E}\left[G_t|S_{t} = s,A_{t} = a\right] \\
        \upsilon_{\pi}(s) &= \mathbb{E}\left[G_{t}|S_{t} = s\right]
    \end{align*}
    $$
    and imitating equation (5), it has:
    $$
    \mathbb{E}\left[G_{t}|S_{t} = s\right] = \sum_{a}\mathbb{E}\left[G_t|S_{t} = s,A_{t} = a\right]\pi(a|s)
    $$
    i.e.,
    $$
    \upsilon_{\pi}(s) = \sum_{a}q_{\pi}(s,a)\pi(a|s) \quad (7)
    $$
    Comparing equation (7) with equation (6) gives:
    $$
    q_{\pi}(s,a) = \sum_{r}p(r|s,a)r+\gamma\sum_{s^{'}}p(s^{'}|s,a)\upsilon_{\pi}(s^{'}) \quad (8)
    $$
    Equation (7) can also be written as $\upsilon_{\pi}(s^{'}) = \sum_{a^{'}}q_{\pi}(s^{'},a^{'})\pi(a^{'}|s^{'})$, then substituting it into (8) gives:
    $$
    q_{\pi}(s,a) = \sum_{r}p(r|s,a)r+\gamma\sum_{s^{'}}p(s^{'}|s,a)\sum_{a^{'}}q_{\pi}(s^{'},a^{'})\pi(a^{'}|s^{'}) \quad (9)
    $$
    Equations (6) and (9) are called the Bellman Equation.