# TimeAsInput

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pleibers.github.io/TimeAsInput.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pleibers.github.io/TimeAsInput.jl/dev/)
[![Build Status](https://travis-ci.com/pleibers/TimeAsInput.jl.svg?branch=main)](https://travis-ci.com/pleibers/TimeAsInput.jl)
[![Coverage](https://codecov.io/gh/pleibers/TimeAsInput.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pleibers/TimeAsInput.jl)

constraint1: norm(dg/dt)  (möglich klein, nicht zu schnell ändern) -> g(t)-> theta_t
constraint2: norm(d2g/dt2) (wiggliness klein halten)

anfangen mit g= affine transformation
tanh statt relu?
zb Lambda_0*t +h = g