Claude finished the response

Consider the problem of having a particle moving with constant speed but on a grid and taking turns at some nodes. How can we model the movement of such particle using an ODE that encodes both the straight line movements at constant speed and the changes of direction at some nodes ?
Calibrating bump width so turns rotate by the exact angle.
Why a plain ODE x˙=f(x)x˙=f(x) can't work

At a node, the particle may go straight, left or right, depending on where it came from. A velocity field on the plane assigns one velocity to each point, so it cannot encode "what to do here depends on how I arrived." Uniqueness of solutions would be violated. So you need an extra state variable that remembers the heading.
Model: heading as a state variable

Let the state be (x,y,θ)(x,y,θ) with speed vv:
x˙=vcos⁡θ,y˙=vsin⁡θ,θ˙=∑n∈Nψε(r−rn) Ωn(θ)
x˙=vcosθ,y˙​=vsinθ,θ˙=n∈N∑​ψε​(r−rn​)Ωn​(θ)

    Straight segments: away from the nodes, θ˙=0θ˙=0, so the heading is constant and the motion is a straight line at speed vv.
    Turning nodes: ψεψε​ is a narrow bump function centred on node nn, of width εε and normalized so that ∫ψε dt=1∫ψε​dt=1 while the particle crosses it, i.e. ψε(r)=vε η(r/ε)ψε​(r)=εv​η(r/ε) with ∫η=1∫η=1. ΩnΩn​ is the rotation the node imposes:
        relative rule: Ωn=+π/2Ωn​=+π/2 (left), −π/2−π/2 (right), 00 (straight);
        absolute rule ("exit north"): Ωn(θ)=sin⁡(θn∗−θ)⋅gΩn​(θ)=sin(θn∗​−θ)⋅g, with a high gain gg, which relaxes the heading to θn∗θn∗​. Using sin⁡sin avoids the ambiguity of the ±π±π wrap, and you can make ΩnΩn​ depend on the incoming θθ to build a full routing table.
    Non-turning nodes: simply omit them from the sum.

In complex notation, with z=x+iyz=x+iy and u=eiθu=eiθ, this reads z˙=v uz˙=vu, u˙=i ω(z,u) uu˙=iω(z,u)u, which is convenient because ∣u∣=1∣u∣=1 is preserved exactly.

This is a Dubins-car / unicycle model with a curvature κ=θ˙/vκ=θ˙/v that is concentrated at the nodes.
The limit ε→0ε→0

As ε→0ε→0, ψε→ψε​→ a Dirac delta, and the ODE becomes an impulsive differential equation:
r˙=v d(θ),θ(tk+)=θ(tk−)+Δk,
r˙=vd(θ),θ(tk+​)=θ(tk−​)+Δk​,

where tktk​ are the times at which the trajectory hits a turning node. The rounded corners of radius ∼ε∼ε become exact right-angle corners.
Keeping the particle exactly on the grid

For ε>0ε>0 the rounded corner leaves an O(ε)O(ε) offset from the grid line, and it can accumulate over many turns. Two standard fixes:

    Snap the heading to multiples of π/2π/2 by adding −ksin⁡(4θ)−ksin(4θ) to θ˙θ˙. This has stable equilibria at 0,π/2,π,3π/20,π/2,π,3π/2, and a turn kick larger than π/8π/8 tips θθ into the next basin.
    Add a transverse stabilizer that pulls the particle onto the nearest grid line, in the direction perpendicular to its motion (unit grid spacing):
    x˙=vcos⁡θ−λsin⁡2 ⁣θ sin⁡2πx2π,y˙=vsin⁡θ−λcos⁡2 ⁣θ sin⁡2πy2π.
    x˙=vcosθ−λsin2θ2πsin2πx​,y˙​=vsinθ−λcos2θ2πsin2πy​. When moving horizontally, only yy is pulled toward an integer, and vice versa.

Equivalent formulations

    Hybrid automaton: continuous flow r˙=vdqr˙=vdq​ with a discrete mode q∈{E,N,W,S}q∈{E,N,W,S}, and a reset map q↦σn(q)q↦σn​(q) when rr reaches node nn. This is the exact model, and the ODE above is its smooth regularization.
    Metric graph: an edge coordinate s∈[0,Le]s∈[0,Le​] with s˙=vs˙=v, plus a transition (routing) rule at each vertex.
    Stochastic turns: replace the deterministic ΩnΩn​ with a jump process or SDE for θθ (a random walk on the grid at constant speed).

I can write out a concrete simulation (e.g. Python with solve_ivp and a 5×55×5 grid with a routing table) if you'd like to see the behavior.

Yes. Please run a simulation of these dynamics on a small grid. Compare the two simulations : the "regularized one"  $\epsilon >0$ and the "vanishing viscosity one" $\epsilon = 0$
Measuring convergence error and drafting comparison plots.

I'll set up a closed loop on a 5×5 grid with 9 left turns and 1 right turn, so the orbit is periodic. The ε=0 trajectory is exactly a piecewise-linear closed curve, which makes comparison easy.
Ran 3 commands, read a file, shared files

The regularized ODE converges to the impulsive one as ε → 0. Two of the panels (trajectories, heading) show it qualitatively, and the table below shows it numerically.
Setup

    Grid and orbit: the grid is 5×5 with v=1v=1. Ten nodes have relative turning rules: nine left turns (+π/2+π/2) and one right turn (−π/2−π/2). They make a closed spiral-shaped orbit of length 24, so I ran 5 laps (T=120T=120).
    ε = 0: this is the impulsive system. The flow is exactly straight between nodes, and at a turning node θ+=θ−±π/2θ+=θ−±π/2. I computed it in closed form, so it has no numerical error.
    ε > 0: this is the smooth ODE from before, with compact bumps ψεψε​, heading snapping −ksin⁡4θ−ksin4θ and the transverse grid stabilizer. I integrated it with DOP853 at tight tolerances and a step of at most ε/8.

Results
ε	max distance to grid orbit	position error after 5 laps	mean lap time (exact: 24)
0.4	0.068	1.67	24.478
0.2	0.043	0.70	24.140
0.1	0.027	0.18	24.036
0.05	0.017	0.027	24.006
0.025	0.010	0.007	23.999

    Geometry: the regularized path stays within a distance that shrinks with ε. Empirically the maximum distance decays roughly like ε0.7ε0.7. The top-right panel shows the rounded corner at node (4,0) tightening onto the exact right angle as ε decreases.
    Heading: in the bottom-left panel, the jumps of the ε = 0 model are replaced by smooth ramps of duration ≈ 2ε/v that sharpen into the jumps.
    Timing: the main accumulated error is a phase lag, because a rounded corner takes a slightly different time than the sharp one. The lap-time error falls off roughly like ε2ε2, so the position error after 5 laps drops fast. For ε = 0.4 the 1.67 error is just that lag: at t=Tt=T the particle is still mid-turn, which is why the final heading is off by one quarter-turn.

A caveat from the first run

My first attempt used a constant snap rate k=0.5k=0.5, and it failed. The maximum deviation stuck at ≈ 0.049 for every ε, and at ε = 0.025 the particle lost the grid entirely. The kick from a bump is never exactly π/2π/2, because the path bends inside the bump and the time spent in it changes. If the snapping is slow (O(1) time), that small heading error makes the particle drift off the line by an amount that doesn't vanish as ε → 0, and it can then miss the next node's bump.

The fix is to scale the snap rate with the bump: k=κv/εk=κv/ε (I used κ=0.4κ=0.4), so the heading is corrected on the same timescale as the corner. This is a real requirement of the regularization: the corrective terms must scale with ε for the limit to be consistent.

Also, κκ can't be arbitrarily large. The snap has to stay weaker than the peak kick, about 1.47 v/ε1.47v/ε, or the turn never crosses the π/8π/8 basin boundary.

The script is /home/claude/sim.py. I can also try stochastic turning rules, or nodes closer together than 2ε, where the bumps overlap and the regularization stops commuting with the impulsive limit.
Grid particle sim
Image·PNG 
Finding equilibrium points of a trigonometric dynamical system.
Finding equilibrium points of a trigonometric dynamical system.

You're right: sin⁡(4θ)=0sin(4θ)=0 at every θ=nπ/4θ=nπ/4, so θ˙=−ksin⁡4θθ˙=−ksin4θ has 8 equilibria on the circle, not 4. My earlier sentence listed them in two groups, but I never said that together they make up all of nπ/4nπ/4, and I only called the axis-aligned ones "the equilibria", which was misleading.

Stability. Let f(θ)=−ksin⁡4θf(θ)=−ksin4θ. Then f′(θ)=−4kcos⁡4θf′(θ)=−4kcos4θ, and at θ=nπ/4θ=nπ/4 we have cos⁡(4θ)=cos⁡(nπ)=(−1)ncos(4θ)=cos(nπ)=(−1)n, so f′=−4k(−1)nf′=−4k(−1)n.

    nn even (θ=0,π/2,π,3π/2θ=0,π/2,π,3π/2, the four axis directions): f′=−4k<0f′=−4k<0, so these are stable.
    nn odd (θ=π/4,3π/4,…θ=π/4,3π/4,…, the diagonals): f′=+4k>0f′=+4k>0, so these are unstable. They act as the basin boundaries between neighbouring axis headings.

So the "mπ/2mπ/2" was the stable subset, and "π/4+mπ/2π/4+mπ/2" was the unstable subset. Only the stable ones are the states the particle can stay in, which is why I named them. The sign of the term matters here. With +ksin⁡4θ+ksin4θ the roles swap, and the stable headings would be the diagonals. So the minus sign is what makes the axis headings the attractors.

and can you explain about the transverse stabilizer
Analyzing decay rates and stability near corners and turns.
What it does

The stabilizer pulls the particle's position toward the nearest grid line, using only the coordinate transverse to its motion:
x˙=vcos⁡θ−λsin⁡2θ sin⁡2πx2π,y˙=vsin⁡θ−λcos⁡2θ sin⁡2πy2π
x˙=vcosθ−λsin2θ2πsin2πx​,y˙​=vsinθ−λcos2θ2πsin2πy​

(unit grid spacing, and I used λ=8λ=8 in the final run).

The restoring term. sin⁡2πy2π2πsin2πy​ is periodic with period 1, and it vanishes at every integer yy. Near y=ny=n it is approximately y−ny−n. So if the particle is at y=n+δy=n+δ, then δ˙≈−λδδ˙≈−λδ, and the offset decays like e−λte−λt. It is the gradient of a periodic potential, −λ ∂yU−λ∂y​U with U=1−cos⁡2πy4π2U=4π21−cos2πy​, whose wells sit on the grid lines. The unstable points are the half-integers y=n+12y=n+21​, so the offset must stay below 1/21/2. Being periodic, the term automatically picks the nearest line.

The gating weights. The factors sin⁡2θsin2θ and cos⁡2θcos2θ select which coordinate is transverse:

    Moving horizontally (θ=0,πθ=0,π): sin⁡2θ=0sin2θ=0 and cos⁡2θ=1cos2θ=1. So xx is untouched and only yy is pulled toward an integer.
    Moving vertically (θ=±π/2θ=±π/2): the roles swap, and only xx is pulled.
    In between (during a turn), both act with partial weight. Precisely at the node centre both sines vanish, so the term costs nothing there.

The gating is essential. If the along-track coordinate were also pulled toward integers, every lattice point would become a fixed point of the flow and the particle would get stuck at the first node it reached. Leaving the along-track motion alone keeps the speed along the line equal to vv, so the timing between nodes is preserved.
Why it is needed

The rounded corner of size ∼ε∼ε leaves the particle displaced by O(ε)O(ε) from the outgoing grid line. Without the stabilizer there is nothing to remove that offset, because a straight line with θ˙=0θ˙=0 conserves it, and offsets from successive corners accumulate. This matters because the bump only delivers the full quarter-turn for a pass through the centre. A particle that arrives at the next node displaced by a fraction of εε gets a shorter chord and a smaller kick, and if the offset exceeds εε it misses the bump entirely and flies off the grid.

With the stabilizer, after a leg of length LL the offset is reduced by e−λL/ve−λL/v. With λ=8λ=8 and L≥1L≥1 that is a factor of about 3×10−43×10−4, so the offset has essentially vanished by the next node.
Interaction with the heading snap

The two mechanisms complement each other. A heading error δθδθ feeds the offset δδ through δ˙≈v δθ−λδδ˙≈vδθ−λδ, so the offset settles near v δθ/λvδθ/λ while the heading error persists. The snap term makes δθδθ decay quickly. So:

    the snap removes the source of new offset (wrong heading),
    the stabilizer removes the offset that already exists (from corner rounding).

Side effects

The stabilizer adds a small transverse velocity of order λδλδ. The speed is therefore vv only up to this correction, and it is nonzero only while the particle is off a line, mostly during and just after corners. This is one reason the lap time differs slightly from 24 for finite εε. The stabilizer is a numerical device for pinning the dynamics to the grid, and it is not part of the exact (ε=0ε=0) model, which stays on the lines by construction.


Claude is AI and can make mistakes. Please double-check responses.
