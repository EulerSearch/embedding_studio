TEST_INPUT_TEXTS = [
    "query: Example text to be tokenized and input into the model.",
    "Example text to be tokenized and input into the model.",
    """I get no kick from champagne
Mere alcohol doesn't thrill me at all
So tell me why should it be true
That I get a kick out of brew""",
    "Do not let your fire go out, spark by irreplaceable spark in the hopeless swamps of the not-quite, the not-yet, and the not-at-all. Do not let the hero in your soul perish in lonely frustration for the life you deserved and have never been able to reach. The world you desire can be won. It exists.. it is real.. it is possible.. it's yours.",
    """     
The Fundamental Theorem of Calculus (FTC) is a cornerstone of mathematical analysis that establishes the connection between differentiation and integration, two principal concepts in calculus. This theorem is divided into two parts, which together serve as the backbone for the computational techniques of calculus applied in science and engineering.

**Part 1: The First Fundamental Theorem of Calculus**

The First Fundamental Theorem of Calculus states that if a function \( f \) is continuous on a closed interval \([a, b]\) and \( F \) is the function defined by the integral of \( f \) from \( a \) to \( x \), where \( x \) is in the interval \([a, b]\), then \( F \) is differentiable on the interval \((a, b)\), and its derivative is \( f(x) \). Mathematically, this is expressed as:
\[ F(x) = \int_{a}^{x} f(t) \, dt \]
\[ F'(x) = f(x) \]
This part of the theorem shows how we can compute the derivative of an integral function. It essentially tells us that integration and differentiation are inverse processes: integrating a function and then differentiating the result gives us back the original function within the limits of the integral.

**Part 2: The Second Fundamental Theorem of Calculus**

The Second Fundamental Theorem of Calculus provides an efficient method for evaluating definite integrals. It states that if \( f \) is continuous on the interval \([a, b]\) and \( F \) is any antiderivative of \( f \) on that interval, then the integral of \( f \) from \( a \) to \( b \) is given by:
\[ \int_{a}^{b} f(x) \, dx = F(b) - F(a) \]
This theorem simplifies the computation of definite integrals by reducing the problem to finding the values of an antiderivative at the endpoints of the interval.

**Applications and Implications**

The Fundamental Theorem of Calculus bridges the gap between analysis and algebraic geometry, allowing for the practical computation of areas, volumes, and other quantities defined in geometric or physical terms. It also underpins the methods for solving differential equations, analyzing dynamic systems, and much more.

In summary, the FTC not only provides theoretical insights into the nature of mathematical analysis but also offers a powerful tool for numerical calculation. Its profound impact is evident across various fields of science and engineering, where calculus is used to model, analyze, and solve complex real-world problems.

""",
]
