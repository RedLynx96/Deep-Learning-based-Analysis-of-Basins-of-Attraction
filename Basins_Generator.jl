using DynamicalSystems
using OrdinaryDiffEq:Vern9
using LaTeXStrings
using CairoMakie
using Colors
using ColorSchemes

@inline @inbounds function duffing(u, p, t)
  d = p[1]; F = p[2]; omega = p[3]
  du1 = u[2]
  du2 = -d*u[2] + u[1] - u[1]^3 + F*sin(omega*t)
  return SVector{2}(du1, du2)
end

@inline @inbounds function forced_pendulum(u, p, t)
  d = p[1]; F = p[2]; omega = p[3]
  du1 = u[2]
  du2 = -d*u[2] - sin(u[1])+ F*cos(omega*t)
  return SVector{2}(du1, du2)
end

F = 0.2; d = 0.1; ; ω = 0.5
res = 333

diffeq = (;reltol = 1e-9, alg = Vern9(), maxiters = 1e6)
ds = ContinuousDynamicalSystem(duffing, rand(2), [d, F, ω]; diffeq)
xg = yg = range(-2.2,2.2,length = res)
smap = StroboscopicMap(ds, 2*pi/ω)
mapper = AttractorsViaRecurrences(smap, (xg, yg); sparse = false)
bsn, att = basins_of_attraction(mapper)


cmap = ColorScheme([RGB(1,1,0), RGB(1,0,0), RGB(0,1,0), RGB(0,0,1), RGB(0,1,1), RGB(1,0,1)] )
F = round(F, digits=3)
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1,1], ylabel = L"y", xlabel = L"x", yticklabelsize = 20, xticklabelsize = 20, titlesize = 20 ,ylabelsize = 20, xlabelsize = 20, title = L"\alpha = %$(1), \beta = %$(1), \gamma = %$(F), \omega = %$(ω)")
heatmap!(ax, xg, yg, bsn, colormap = cmap, colorrange = (1,6))
fig
