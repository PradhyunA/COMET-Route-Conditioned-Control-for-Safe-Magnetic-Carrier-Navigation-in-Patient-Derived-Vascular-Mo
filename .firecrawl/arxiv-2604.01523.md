arXiv:2604.01523v1  [cs.RO]  2 Apr 2026

Robust Autonomous Control of a Magnetic Millirobot in In Vitro
Cardiac Flow
[Image: Im7]
[Image: Im46]

1∗
Anuruddha Bhattacharjee, Xinhao Chen1, Lamar O. Mair2, Suraj Raval3,
1∗
Yancy Diaz-Mercado3, Axel Krieger
[Image: Im7]
[Image: Im46]

Abstract— Untethered magnetic millirobots offer significant
potential for minimally invasive cardiac therapies; however,
achieving reliable autonomous control in pulsatile cardiac
flow remains challenging. This work presents a vision-guided
control framework enabling precise autonomous navigation of
a magnetic millirobot in anin vitroheart phantom under
physiologically relevant flow conditions. The system integrates
UNet-based localization, A* path planning, and a sliding mode
controller with a disturbance observer (SMC-DOB) designed
for multi-coil electromagnetic actuation. Although drag forces
are estimated using steady-state CFD simulations, the controller
compensates for transient pulsatile disturbances during closedloop operation. In static fluid, the SMC-DOB achieved submillimeter accuracy (root-mean-square error, RMSE = 0.49
mm), outperforming PID and MPC baselines. Under moderate
pulsatile flow (7 cm/s peak, 20 cP), it reduced RMSE by
37% and peak error by 2.4×compared to PID. It further
maintained RMSE below 2 mm (0.27 body lengths) under
elevated pulsatile flow (10 cm/s peak, 20 cP) and under lowviscosity conditions (4.3 cP, 7 cm/s peak), where baseline
controllers exhibited unstable or failed tracking. These results
demonstrate robust closed-loop magnetic control under timevarying cardiac flow disturbances and support the feasibility of
autonomous millirobot navigation for targeted drug delivery.
[Image: Im30]
[Image: Im31]
[Image: Im46]
[Image: Im47]

\mathbf{A}^{*}

37\%

I. Introduction
[Image: Im46]
[Image: Im47]

[Image: Im31]
Magnetically actuated milli-/microrobots are emerging as
promising tools for targeted therapies in the cardiovascular
system [1]. External magnetic fields enable wireless steering[Image: Im32]
through narrow and tortuous anatomical pathways, such as
[Image: Im33]
blood vessels and cardiac chambers, with high precision [2].
[Image: Im34]
Prior studies have demonstrated[Image: Im46]upstream motion against
blood flow for drug delivery using microrollers [3], swimming in whole blood [4], and autonomous exploration of
[Image: Im47]
vascular networks using magnetic microrobot swarms [5].
These advances highlight the potential of magnetic robots
for treating infective endocarditis [6], in which bacteria infect
arXiv:2604.01523v1  [cs.RO]  2 Apr 2026
cardiac valve regions (Fig. 1), necessitating precise, localized
therapeutic delivery deep within the heart to achieve high
drug concentrations at the infection site.
However, achieving reliable autonomous navigation in

However, achieving reliable autonomous navigation in
cardiac environments remains challenging. Blood flow inside

[Image: Im40]
Fig. 1. Conceptual illustration of targeted cardiac drug delivery using
[Image: Im4]magnetically actuated millirobot to treat infective endocarditis.[Image: Im1]
[Image: Im28] [Image: Im41]
[Image: Im15]
[Image: Im29] [Image: Im42]
[Image: Im7]
[Image: Im46]
[Image: Im47]

[Image: Im15]
[Image: Im29] [Image: Im42]
the heart is fast, pulsatile, and spatially varying [7], creating
[Image: Im30]strong disturbances and model uncertainties that can desta-
[Image: Im7]
bilize magnetic actuation. Several closed-loop control strate-
[Image: Im31]gies have been explored to improve robustness under flow.
[Image: Im43]
Sliding mode control (SMC) with a disturbance observer
[Image: Im44]
[Image: Im32](DOB) has been shown to improve 3D path following for helical microrobots under strong flow disturbances and actuator
saturation [8], while extended-state-observer SMC improved
[Image: Im46]tracking in vascular phantoms [9]. A double-loop SMC
framework with an inner DOB has also been developed for
electromagnetic actuator systems, enabling robust position
[Image: Im47]
and speed tracking under nonlinear model uncertainties [10].
In parallel, learning-based methods, including reinforcement
learning [11], hierarchical deep reinforcement learning [12],
arXiv:2604.01523v1  [cs.RO]  2 Apr 2026
learning-enhanced model predictive control [13], adaptive
learning–based control [14], and deep reinforcement learning–based navigation and flow rejection control [15], [16],
have enabled adaptive path planning and control. These
approaches have demonstrated improved tracking in static
or tubular environments, including flow rejection under constant or sinusoidally varying flow with peak velocities of
approximately 6–10 mm/s [16] and precise tracking in static
fluid without external flow disturbances [14]. While these
results represent important progress toward robust control,
their experimental validation has largely been limited to
relatively low flow velocities and simplified geometries.
Most existing controllers remain limited in their ability

This work was funded by the National Science Foundation’s Foundational
Research in Robotics CAREER program under award number 2144348.
1Laboratory for Computational Sensing and Robotics, Johns Hop-

Most existing controllers remain limited in their ability
to operate reliably under realistic cardiac-like pulsatile flow

1Laboratory for Computational Sensing and Robotics, Johns Hopkins University, Baltimore, MD 21218, USA. Emails:{abhatt27,
xchen254, axel}@jhu.edu
2Division of Magnetic Manipulation and Particle Research,

xchen254, axel}@jhu.edu
2Division of Magnetic Manipulation and Particle Research,
Weinberg Medical Physics, Inc., Rockville, MD 20852, USA.
Email:lamar.mair@gmail.com
3Department of Mechanical Engineering, University of Maryland, Col-

3Department of Mechanical Engineering, University of Maryland, College Park, MD 20742, USA. Emails:{sraval, yancy}@umd.edu
∗Corresponding Authors

---

Fig. 2. (a) Experimental system: (i) Twelve-coil electromagnetic actuation setup surrounding the heart phantom, with a top-mounted camera for vision-based
feedback control; (ii) Enlarged view of the heart phantom showing the internal flow circuit, where a peristaltic pump recreates right atrial and ventricular
flow; (iii) Magnetic millirobot capsule containing an embedded permanent magnet, a drug chamber, and micro-holes for controlled drug release. (b) Finite
element simulations of the electromagnetic system used to characterize magnetic flux density and field gradients under different coil activation patterns.

10\,\times\,10\,\times\,10\,\,\mathrm{c m}^{3}

II. System and Methods

A. Magnetic Robotic System

This work develops a vision-guided autonomous control
framework for a drug-filled magnetic millirobot in anin
vitroheart phantom under physiologically relevant pulsatile
flow. The contributions are: 1) integration of UNet-based
localization and A*-based trajectory planning for closed-loop
navigation in cardiac phantom; 2) a sliding mode controller
with disturbance observer (SMC-DOB) coupled with CFDderived drag estimation to enable robust trajectory tracking
under pulsatile disturbances; and 3) experimental validation of autonomous control under physiologically relevant
pulsatile flow and viscosity variations, with comparative
evaluation against baseline controllers.
II. System and Methods

workspace. The setup consists of twelve planar coils arranged
as four units, each containing three concentric coils of
different radii with axes aligned in thex–yplane. Each coil
is independently driven by six Roboclaw 2×60 AHV motor
drivers powered by 60 V supplies and controlled through
a custom Python interface communicating with a Teensy
microcontroller for pulse-width modulation. This configuration enables the generation of tunable magnetic field
gradients for high-force actuation and precise positioning,
with specifications comparable to previously reported multicoil electromagnetic actuation systems [19].
Finite element simulations (COMSOL Multiphysics) were

Magnetic millirobot design:A capsule-type magnetic
millirobot was developed for targeted drug delivery in the

cardiac environment. Figure 2(a)(iii) shows the robot filled
with a dye solution (simulated drug) and its CAD model.
The robot comprises a plastic body with an internal drug
chamber, an embedded axially magnetized neodymium disk
magnet (D0110-10, SuperMagnetMan), and a release cap for
fluid discharge. The magnet has a 1 mm diameter and length
(grade N50) and generates a numerically estimated pulling
force of 0.37 mN under a 0.43 T/m magnetic field gradient
based on FEA field simulations. The body components were
fabricated using BIO resin on a microArch
printer. The release cap contains seven circular holes (0.3 mm

cardiac environment. Figure 2(a)(iii) shows the robot filled
with a dye solution (simulated drug) and its CAD model.
The robot comprises a plastic body with an internal drug
chamber, an embedded axially magnetized neodymium disk
magnet (D0110-10, SuperMagnetMan), and a release cap for
fluid discharge. The magnet has a 1 mm diameter and length
(grade N50) and generates a numerically estimated pulling
force of 0.37 mN under a 0.43 T/m magnetic field gradient
based on FEA field simulations. The body components were
®
fabricated using BIO resin on a microArch S140 resin 3D
printer. The release cap contains seven circular holes (0.3 mm diameter) arranged in a ring with 0.6 mm center-to-center
spacing, enabling passive drug release via diffusion into the
surrounding fluid. The assembled capsule measures 2.80 mm
in diameter and 7.40 mm in length, carrying up to 22µL
of liquid. These dimensions enable passage through cardiac
structures such as the tricuspid valve and right ventricular
outflow tract (27-33 mm [20]).
Heart phantom with pulsatile flow:A heart phantom

22~\mu\mathrm{L}

Heart phantom with pulsatile flow:A heart phantom
was constructed as anin vitroplatform to evaluate millirobot
navigation under physiologically relevant flow. The phantom
was molded from 2% (w/w) agarose gel to approximate
cardiac tissue elasticity [21] and cast from a cross-sectional
TM
heart model using a silicone mold (Ecoflex 00-20 Fast,
1:1 ratio). The resulting structure represents portions of the
right atrium, right ventricle, and pulmonary trunk, including
narrowed valve-like regions and curved channels near the
tricuspid and pulmonary valves (Fig. 1, Fig. 2(a)(ii)).
Pulsatile flow was generated using a peristaltic pump

tricuspid and pulmonary valves (Fig. 1, Fig. 2(a)(ii)).
Pulsatile flow was generated using a peristaltic pump
circulating glycerin–water mixtures through a closed-loop
circuit connected to the phantom. Under the tested pump settings, the flow exhibited an approximate pulsation frequency
of 3 Hz, within the physiological range observed under
elevated cardiac activity. Flow rates of 180 and 250 mL/min
produced peak velocities of approximately 7–10 cm/s, consistent with physiological right atrial flow during resting
conditions [22]. Fluid viscosity was adjusted by varying the
glycerin–water ratio (40:60 to 65:35) to achieve 4.3 and
◦
20 cP at 23 C, representing near-healthy and elevated blood
viscosity levels, respectively [23]. These conditions were
used to evaluate controller robustness under varying fluid
resistance and pulsatile disturbances.

23^{\circ}\mathbf{C}

B. Modeling and State Estimation
Magnetic actuation dynamics:A magnetic field with
spatial gradients exerts both force and torque on a robot
containing an embedded permanent magnet. The magnetic
forceFmand torqueTare given by
T
Fm=m ∇B,T=m×B,(1)

\left[\begin{matrix}{F_{m x}}\\ {F_{m y}}\\ {\omega_{o}B_{x}}\\ {\omega_{o}B_{y}}\\ \end{matrix}\right]=\left[\begin{matrix}{{m}^{T}\left[\frac{\partial\widetilde{B_{1}}}{\partial x}\cdots\frac{\partial\widetilde{B_{8}}}{\partial x}\right]}\\ {{m}^{T}\left[\frac{\partial\widetilde{B_{1}}}{\partial y}\cdots\frac{\partial\widetilde{B_{8}}}{\partial y}\right]}\\ {\widehat{B_{1}}{\ }\ vdots\ \ \ \

\boldsymbol{F}_{m}=\boldsymbol{m}^{T}\boldsymbol\nabla{B},~~\boldsymbol{T}=\boldsymbol{m}\times\boldsymbol{B},

F_{m}

The electromagnetic actuation system comprises twelve
planar coils arranged for in-plane (x–y) control. In a lowviscosity fluid, the robot rapidly aligns with the field and
requires only low field strength for translation; therefore,
rotational dynamics are neglected, and eight coils are used for
planar control. The mapping from coil currents to magnetic
forces and field components is expressed as
 h i
  T ∂Bf ∂Bf
m ···  

∇B,T=m×B,(1)

wheremis the magnetic dipole moment,∇Bis the magnetic field gradient, andBis the magnetic flux density, which
scales linearly with coil current.
The electromagnetic actuation system comprises twelve

B_{x}

B_{y}

\omega_{o}

C\;=\;A I

field components, andωois a unit conversion factor. This
can be written compactly asC=AI, whereAis the
current-to-field mapping matrix obtained from finite element
analysis [24].
To compute the coil currents for a desired control vector

To compute the coil currents for a desired control vector
C,Ais decomposed using singular value decomposition
(SVD),
T
A=UΣW,(3)

i,\ I_{i}

\ \ A U U\Sigma^{T},

,(5)

whereUandWare orthogonal matrices, andΣis diagonal
with singular values. Then the input currents are obtained as

,(6)

I=W\Sigma^{+}U^{T}C,

+
whereΣ is the Moore–Penrose pseudoinverse.
During motion in a fluid, the robot experiences viscous

During motion in a fluid, the robot experiences viscous
drag,
F c V

\Sigma^{+}

\left[\begin{matrix}{F_{d x}}\\ {F_{d y}}\\ \end{matrix}\right]=\left[\begin{matrix}{c_{t}V_{x}}\\ {c_{t}V_{y}}\\ \end{matrix}\right],

whereVxandVyare flow velocities. The translational
damping coefficient for an ellipsoid is [25]

V_{\alpha}

V_{y}

whereµdbeing the dynamic viscosity of the fluid, andL
andrare the robot length and radius.
CFD-based flow field estimation:To estimate the flow

\mu_{d}

andrare the robot length and radius.
CFD-based flow field estimation:To estimate the flow
field in the heart phantom, steady-state laminar flow was
simulated in a computational fluid dynamics (CFD) software
(ANSYS Fluent) using a simplified heart phantom geometry
(Fig. 3). The model consisted of two straight inlet channels
(D= 3mm,L= 85mm) feeding a central chamber
and a single outlet, with all lumen walls defined as no-slip
boundaries. The computational mesh (Fig. 3(a)) employed a
patch-conforming tetrahedral core with prism-layer inflation
5
along the walls (∼5.4×10 elements, 12 layers, first
layer0.015mm, growth rate 1.2). The working fluid was
−3
a Newtonian glycerin–water mixture (ρ= 1113 kg m,
µ= 0.0043 Pa s), with velocity-inlet and pressure-outlet
boundary conditions. A pressure-based SIMPLE solver with
second-order schemes was used, and convergence was ac-
−6
cepted when residuals fell below10.
For inlet flow rates ofQ= 250and180mL/min, the

L_{e}\approx0.05\,\mathrm{R e}\,D

(\sim\,\,5.4\,\times\,10^{5}

(\mu=0.020\,\mathrm{P a}{\cdot}{mathsf s))}

10^{-6}

cepted when residuals fell below10.
For inlet flow rates ofQ= 250and180mL/min, the
average inlet velocities wereU≈0.295and0.212m/s, corresponding to Reynolds numbersRe≈227and164. Using
the laminar entrance length estimateLe≈0.05 ReD[26],
the required entrance lengths (34 and 25 mm) were much
shorter than the actual inlet length (85 mm), confirming fully
developed inlet profiles. At higher viscosity (µ= 0.020Pa·s),
Redropped below 50 with similarly short entrance lengths.
The converged steady-state velocity field on the actuation
plane (Fig. 3(b)) was exported, and only the in-plane (x-y)
components(Vx,Vy)were used for drag estimation and control. Figure 3(b) shows the velocity contour at250mL/min
inlet flow rate, producing peak velocities up to 10 cm/s within
the phantom near the inlet-2 region.
Vision-based localization and pose estimation:Au-

\mu\;=\;0.0043

---

Fig. 3. (a) Computational mesh of the heart phantom model with two flow
inlets and an outlet and (b) velocity magnitude contour plot for 4.3 cP fluid
at the plane of robot control (i.e.,z= 8 mm, wherez= 0 mm is the plane
passing through the centerline of the inlets and outlets) and0.295cm/s flow
speed at each inlet.

z=8

z=0

of the millirobot’s position and orientation. We developed
a vision-based localization framework using a modified
UNet [27] that performs both semantic segmentation and
keypoint detection, enabling simultaneous estimation of the
millirobot mask and its tip, centroid, and tail keypoints.
Unlike prior work that used UNet solely for segmentation
of magnetic needles in surgical settings [28], this model
was trained specifically for a drug-filled millirobot navigating
inside the heart phantom.
A dataset of 2,005 labeled images was collected during

inside the heart phantom.
A dataset of 2,005 labeled images was collected during
millirobot experiments in the heart phantom. Segmentation masks were generated using a SAM2-based automated
pipeline [29], and keypoints were annotated manually. The
model was trained using cross-entropy loss for segmentation
and L1 loss for keypoints (L=LCE+ 0.5LL1) with
−3
Adam (10), batch size 3, ReduceLROnPlateau scheduling,
◦
and early stopping. Augmentations included flips and90
rotations. During inference, cropped frames were resized
to572×572and processed by the network. The centroid
keypoint defined position, and the tip–tail vector defined
orientation. This orientation information is required to align
the applied magnetic field with the robot’s magnetic moment,
ensuring that the generated magnetic force produces the
intended translational motion. The model ran at 18 fps
on an Nvidia GeForce RTX-4070-Ti and achieved98.7%
precision,98.3% recall, and97.1% IoU, enabling reliable
real-time pose estimation under pulsatile flow.

(\mathcal{L}\ =\ \mathcal{L}_{\mathrm{C E}}\,+\,0.5\,\mathcal{L}_{\mathrm{L l}})

Path planning and trajectory generation:An imagebased path planning pipeline using the A* algorithm was
developed to generate reproducible navigation trajectories. A
top-view image of the heart phantom (1020×1020 pixels,

top-view image of the heart phantom (1020×1020 pixels,
≈92×92 mm2) was used
the navigable canal was manually outlined to create a binary mask. A Euclidean distance transform was applied to
compute per-pixel wall clearance,
clearance constraint defined the feasible core region. A cost
map combining uniform motion cost with a clearance-based
centerline bias (wclear=
region. User-selected start and end points were projected to
the nearest feasible pixels, and an 8-connected A* search

top-view image of the heart phantom (1020×1020 pixels,
used as the planning domain, where
the navigable canal was manually outlined to create a binary mask. A Euclidean distance transform was applied to
clearance, and a 5 mm minimum
clearance constraint defined the feasible core region. A cost
map combining uniform motion cost with a clearance-based
0.5) was constructed within this
region. User-selected start and end points were projected to
the nearest feasible pixels, and an 8-connected A* search

(10^{-3})

90^{\circ}

C. Planning and Control Framework

\sqrt{2}

572\,\times\,572

(1020\!\times\!\dot{1}020

\ \mathrm{A}^{*}

{approx92}{\times}92\,\mathrm{{m m^{2}}}

\ {mathrm A^{*}}

8{\cdot}

Fig. 4. A* planned trajectory (yellow) within the user-defined canal mask
with 5 mm wall clearance and 50% centerline bias. Start (blue star) and end
(red star) points are connected through resampled waypoints (magenta), and
the green spline indicates the user-selected canal boundary.
√

\mathrm{A}^{*}

√
was performed with diagonal and straight step costs of 2
and 1, respectively. The resulting path was smoothed using
a moving-average filter (window size= 7) and resampled
into 10 evenly spaced waypoints. These were converted to
metric coordinates using the pixel-to-millimeter ratio and
provided to the controller as the desired trajectory (Fig. 4).
This trajectory was computed offline and remained fixed
during experiments, providing a consistent reference path
for evaluating controller performance rather than dynamic
replanning. The start and end points were selected to mimic
navigation from the inlet toward the pulmonary valve region
while traversing the tricuspid valve area, representing a
clinically relevant path for targeted drug delivery (see Fig. 1
and Fig. 2).
Sliding mode controller with disturbance observer:

\lambda

and Fig. 2).
Sliding mode controller with disturbance observer:
To enable reliable autonomous control under pulsatile flow
disturbances and model uncertainties, we developed a sliding
mode controller with an integrated disturbance observer
(SMC–DOB). While inspired by the general SMC–DOB
framework in [10], which was implemented for a different
6-coil electromagnetic system, the controller in this work
was newly formulated for thein vitrocardiac environment.
The sliding surface, switching term, and disturbance observer
were designed to account for the millirobot’s hydrodynamic
drag, electromagnetic actuation characteristics, and flow disturbances. The controller computes a planar force command
⊤
Fdes= [Fx,Fy] to drive the robot from its current position
⊤ ⊤
x= [x,y] to the desired positionxd= [xd,yd], which is
mapped to coil currents using Eq. (2).
The sliding surface is chosen based on the tracking error

\mathbf{x}_{d}=[x_{d},y_{d}]^{\top}

The sliding surface is chosen based on the tracking error
e=xd−xand its derivative e˙ = x˙d− x˙, described as

\mathbf{F}_{\mathrm{d e s}}=[F_{x},F_{y}]^{\top}

whereγ >0andλ >0determine the convergence rate and
damping of the sliding dynamics. In our system,γandλare
set to 2 and 1, respectively, for a stable response.
The control law for the desired force includes three terms

\mathbf{s}=\gamma\dot{\mathbf{e}}+\lambda\mathbf{e},

\lambda>0

The control law for the desired force includes three terms
and is expressed by

---

\hat{\mathrm{d}}

where dˆis the disturbance estimate from the disturbance
observer (DOB),ueqis a model-based equivalent term, and
uswis a nonlinear switching term enforcings→0.
The DOB estimates unmodeled drag and flow forces using

\mathbf{u}_{\mathrm{e q}}

\mathbf{u}_{\mathrm{s w}}

\mathbf{s}\rightarrow\mathbf{0}

The DOB estimates unmodeled drag and flow forces using
two low-pass filters:

\dot{\mathbf{p}}=-\frac{1}{\eta}\mathbf{W}_{p}\big(\mathbf{p}-\dot{\mathbf{e}}\big),

(p− e˙),(9)

\dot{\mathbf{r}}=-\frac{1}{\eta}\mathbf{W}_{r}(\mathbf{r}-\mathbf{F}_{\mathrm{p r e v}}),

,∥B∥= 1.(16)

),(10)

wherep,r∈R2 are filter states,η >0sets the filter
2×2
bandwidth,Wp,Wr∈R are diagonal weight matrices,
andFprevis the previously applied force. The estimated
disturbance is
1 W

\mathbf{p},\mathbf{r}\ \in\ \mathbb{R}^{2}

\eta\,>\,0

\mathbf{W}_{p},\mathbf{W}_{r}\in\mathbb{R}^{2\times2}

\mathbf{F}_{\mathrm{p r e v}}

\hat{\mathbf{d}}=m\Big(\frac{1}{\eta}\mathbf{W}_{p}(\mathbf{p}-\dot{\mathbf{e}})\Big)-\mathbf{C}(\dot{\mathbf{x}},\mathbf{V})-\mathbf{r},

⊤
wheremis the robot mass,V= [Vx,Vy] is the estimated
flow velocity, andC(x˙,V) =ct(x˙ +V)represents the
modeled viscous and flow-induced drag (see Eq. (5)).
The equivalent term compensates for nominal dynamics

−C(x˙,V)−r,(11)

\mathbf{V}=[V_{x},V_{y}]^{\top}

\mathrm{C}(\dot{\bf{x}},\mathrm{V})\;=\;c_{t}(\dot{\bf{x}}+\mathrm{V})

The equivalent term compensates for nominal dynamics
and shapes the closed-loop error response:

\mathbf{u}_{\mathbf{e q}}=m\mathbf{K}_{3}^{-1}\big(\mathbf{K}_{1}\dot{\mathbf{e}}+\mathbf{K}_{2}\mathbf{e}\big)+m\ddot{\mathbf{x}}_{d}-\mathbf{C}(\dot{\mathbf{x}},\mathbf{V}),

2×2
whereK1,K2,K3 ∈R are positive gains (scaled by
fluid viscosity) and x¨dis the desired acceleration (set to zero
when unknown). The switching term drives the system to the
sliding surface while mitigating chattering:
s

\mathrm{K}{_11,\mathrm{K}_{2},\mathrm{K}_{3}}\,\in\,\mathbb{R}^{2\times2}

\dot{\bf x}_{d}

\mathbf\{u{}}}_{{\mathrm{s w}}}K=__{4}\operatorname{t a n h}\!\left(\frac{\mathbf{s}}{\phi}\right),

K_{4}\,>\,0

\phi\,>\,0

Lemma (SMC-DOB stability):Let the lumped disturbance
d(t)be bounded and Lipschitz continuous, and let the DOB
produce a bounded estimation error d˜=d− dˆsatisfying
∥d˜∥ ≤δ, whereδ ≥0denotes the maximum
max max
estimation error bound. IfK4 > δmax, then the error is
ultimately uniformly bounded. Furthermore, ifδmax→0,
thens(t)→0and consequentlye(t)→0asymptotically.
Proof:Follows from the Lyapunov candidate

\tilde{\bf{d}}\,=\,{\bf{d}}-\hat{\bf{d}}

\|\tilde{\bf d{}}}\|\:\:\leq\:\:\delta_{\operatorname*{m a x}}

K_{4}\ >\ \delta_{\operatorname*{m a x}}

\delta_{\mathrm{m a x}}~\geq~0

\dot{V}\ \leq\ 0

· e˙,(14)

\mathbf{s}(t)\!\ \!\ \ \ !to!mathbf{0}

\mathbf{e}(t)\!\to\!\mathbf{0}

Proof:Follows from the Lyapunov candidate
1 ⊤ ˜⊤˜(κ >0)and the switching controller
V= s s+ d d
2 21κ
in Eq. (13). IfK4 > δmax, there exists a ball whose radius
depends onϕsuch that V˙≤0for alls(t)outside this
ball. As such,∥s(t)∥→O(ϕ). If the DOB converges, then
the sliding dynamics converge to the origin, and the error
converges to zero asymptotically [30].
PID baseline controller:For baseline comparison, a

,(18)

\theta_{d}.

K_{4}>\delta_{\mathrm{m a x}}.

V,(15)

\textstyle V=\frac{1}{2}\ {\tilde\mathtt{s}}^{\top}\mathtt{s}+\frac{1}{2\kappa}{\tilde{\mathtt{d}}}^{\top}{\tilde{\mathtt{d}}}\ (\kappa>0)

whereVis the estimated flow velocity andctis the
translational damping coefficient from Eq. (6). Orientation is
controlled independently using a unit magnetic field aligned
with the desired heading angleθd,

\begin{array}{r}{\mathbf{F}_{\mathrm{d e s}}=\mathbf{F}_{\mathrm{P I D}}-\mathbf{F}_{d},\ \ \ \ \mathbf{F}_{d}=c_{t}\mathbf{V},}\end{array}

\mathbf{B}=\left[\begin{matrix}{\sin\theta_{d}}\\ {\cos\theta_{d}}\end{array}\right],\qquad\|\mathbf{B}\|=1.

The desired forceFdesand fieldBare converted to coil
currents using Eq. (2).
MPC baseline controller:A model predictive control

\mathbf{F}_{\mathrm{d e s}}

MPC baseline controller:A model predictive control
(MPC) framework was implemented as a baseline to compute
the desired forceFdesand magnetic fieldBdes. The robot
dynamics are modeled as

\ \ \ M cdot dot\ensuremath{\mathbf{v}}(t)+\mathbf{F}_{d r a g}=\mathbf{F}_{\mathrm{d e s}}(t),

whereMis the robot mass andFdragis the drag force from
Eq. (5).
The prediction horizon was set toN= 20with a time

\mathbf{F}_{d r a g}

The prediction horizon was set toN= 20with a time
step of∆t= 40ms. The MPC cost function is

\Delta t=40

J=\sum_{k=0}^{N-1}\left(e_{k}^{T}\mathbf{Q}e_{k}+F_{k}^{T}\mathbf{R}F_{k}\right)+e_{N}^{T}\mathbf{P}e_{N},

whereekandFkdenote the position error and desired force
at stepk, andQ,R, andPare weighting matrices.
The optimization problem was solved using the Multi-

e_{k}

F_{k}

The optimization problem was solved using the Multifrontal Massively Parallel Sparse Direct Solver (MUMPS).
The first optimal force input was combined with the magnetic
fieldBfrom Eq. (16) and mapped to coil currents using
Eq. (2).

III. Experiments and Results

A. Baseline Trajectory Tracking in Static Fluid

We conducted experiments to evaluate the trajectory tracking
performance of the proposed sliding mode controller with
disturbance observer (SMC-DOB) in the heart phantom.
Control difficulty was systematically varied by changing two
disturbance factors: fluid viscosity and flow strength. Tests
were performed in high-viscosity (20 cP) and low-viscosity
(4.3 cP) fluids, each under static (no flow) and pulsatile flow
with peak in-phantom velocities of 7 cm/s and 10 cm/s. These
conditions reflect physiologically relevant ranges, as 4.3 cP
and 20 cP correspond to near-healthy and elevated blood
viscosities [23], and 7–10 cm/s falls within the resting right
atrial flow range [22]. Proportional–integral–derivative (PID)
and model predictive control (MPC) were implemented as
baseline controllers. The following subsections first present
baseline performance in static fluid, then evaluate controller
performance under increasing disturbance levels, and finally
assess the contribution of the disturbance observer through
ablation experiments.

---

TABLE I

Baseline trajectory tracking in static fluid. Values represent mean
±standard deviations across three trials per condition.

\pm

| Controller | Viscosity(cP) | RMSE(mm) | P95(mm) | Max(mm) |
| --- | --- | --- | --- | --- |
| PID | 20 | 0.82±0.02 | 1.78±0.05 | 2.70±0.15 |
| MPC | 20 | 1.57±0.04 | 2.77±0.14 | 4.40±0.38 |
| SMC-DOB | 20 | 0.49±0.01 | 0.86±0.03 | 1.64±0.21 |
| PID | 4.3 | failed to complete trajectory |  |  |
| MPC | 4.3 | failed to complete trajectory |  |  |
| SMC-DOB | 4.3 | 1.05±0.02 | 1.90±0.05 | 3.21±0.23 |

0.82\!\pm\!0.02

1.78{\pm}0.05

1.57{\pm}0.04

2.70{\pm}0.15

2.77{\pm}0.14

\mathbf{0.49\pm 0.01}

4.40{\pm}0.38

\ \mathbf{0.86\ 003

\\mathbf{1.64\\pm 0.21}

{\bf1.05\pm0.}{\bf02}

\ \mathbf1.90{\pm}0.05

\mathbf{3.2\dot{1}\pm0.23}

◦
0.5 mm/s, with the field heading fixed at45 relative to the
x-axis.
For the PID controller, the gainsk,k, andk were

45^{\circ}

For the PID controller, the gainskp,ki, andkdwere
−2 −1 −3 −1 −1
set to5×10 N·m,5×10 N·s ·m, and8×
−3 −1
10 N·s·m, respectively. For SMC-DOB, the switching
gainK4, disturbance filter bandwidthη, and boundary-layer
−4 −2 −2 −1
thicknessϕwere2.0×10 N·s,6.0×10 s, and9.0×
−3
10 m, respectively. Tracking performance was quantified
using root mean square error (RMSE), 95th percentile error
(P95), and maximum error over three trials per condition
(Table I).
In 20 cP fluid, SMC-DOB achieved the highest accuracy,

k_{p}

k_{i}.

5\,\times\,10^{-2}\,\,\ \ \mathbf{N}{\cdot}\mathbf{m}^{-1},\ 5\,\times{\,\dot{\,}}}\&{{}\,{\bf N}{\cdot}{\bf\dot{\,}s{^{-1}}}\,{{\bf\cdot}}{\bf m}^{-1}

8~\times

10^{-3}\ {\mathrm{N}}{\cdot}{\mathrm{s}}{\cdot}{\mathrm{m}}^{-1}

K_{4}

\eta,

\phi

2.0\times10^{-4}\,\mathrm{N}{\cdot s^{-2}},6.0\times10^{-2}\,s^{-1}

9.0\times

10^{-3}

In 20 cP fluid, SMC-DOB achieved the highest accuracy,
with a mean RMSE of 0.49 mm (0.066 body lengths) and
maximum error of 1.64 mm. PID also maintained submillimeter accuracy, while MPC exhibited larger deviations.
In 4.3 cP fluid, both PID and MPC failed to complete the
trajectory due to increased disturbance, whereas SMC-DOB
successfully tracked it with a mean RMSE of 1.05 mm.

B. Tracking Performance under Moderate Flow Conditions

Trajectory tracking was next evaluated under moderate flow
conditions (viscosity: 20 cP; peak in-phantom velocity:
7 cm/s) to compare the proposed SMC–DOB controller with
a conventional PID controller. Fluid-drag forces were estimated using time-averaged CFD velocity fields and computed
using Eq. (5) and Eq. (6). The PID controller used the same
gains as in static tests, except for an increased derivative gain
−2 −1
kd= 2.0×10 N·s·m to provide additional damping
against flow-induced oscillations. The SMC–DOB controller
−4 −2
was re-tuned withK4 = 3.0×10 N·s,η= 12.0×
−2 −1 −3
10 s, andϕ= 18.0×10 m. Compared to static conditions, these parameters were increased to compensate for
higher drag, accelerate disturbance estimation, and suppress
chattering while maintaining stability.
As shown in Fig. 5, SMC–DOB maintained a tighter tra-

k_{d}=2.0\times10^{-2}\ {\mathbb N}{\cdot}{\mathbb s}{\cdot}{\mathfrak m}^{-1}

4.36{\pm}0.21\ \mathrm{m m}.

1.00{\pm}0.02

10^{-2}s\^{-1}

t\!\approx\!50\ \S

[Image: Im1]
(a) PID Controller

[Image: Im2]
(b) SMC-DOB Controller

(c) 
10

Fig. 5. Trajectory tracking under moderate flow conditions (20 cP; 7 cm/s
peak speed). Representative video snapshots show the millirobot’s path,
color gradient from start (ti) to end (tf), inside the heart phantom using
(a) PID and (b) SMC–DOB controllers. (c) Tracking error over time.
SMC–DOB maintained a tighter trajectory tracking and reduced error spikes
compared to PID, demonstrating improved robustness to flow disturbances.

(t_{f})

(t_{i})

We next evaluated SMC-DOB robustness under stronger
flow disturbances by increasing flow speed while keeping
viscosity constant at 20 cP. Pulsatile flow was generated
at peak in-phantom velocities of 7 cm/s (moderate flow,
180 mL/min) and 10 cm/s (elevated flow, 250 mL/min). The
millirobot followed the same inlet-to-outlet trajectory.
Figure 6(a) shows overlaid trajectories, and Fig. 6(b) shows

\eta

y=-10

D. Robustness to Viscosity Changes

loss of positional hold.
To compensate, the controller was locally retuned when
entering this region: the coil current limit was increased by
1.78×, and the disturbance filter bandwidthηand boundary
layer thicknessϕwere increased by1.75×and1.5×, respec-

layer thicknessϕwere increased by1.75×and1.5×, respectively, to improve disturbance
stability. With these adjustments, the robot completed the
trajectory. Across three trials, SMC-DOB maintained RMSE
below 2 mm (0.27 body lengths) at 10 cm/s, with a maximum
error of8.2±0.2mm, demonstrating reliable autonomous
control under substantially
bances than previously reported experimental conditions.

layer thicknessϕwere increased by1.75×and1.5×, respecdisturbance rejection while maintaining
stability. With these adjustments, the robot completed the
trajectory. Across three trials, SMC-DOB maintained RMSE
below 2 mm (0.27 body lengths) at 10 cm/s, with a maximum
error of8.2±0.2mm, demonstrating reliable autonomous
stronger pulsatile flow disturbances than previously reported experimental conditions.

---

(a)

(b)

Fig. 6. Trajectory tracking under moderate flow (20 cP; 7 cm/s peak
velocity) and elevated flow (20 cP; 10 cm/s peak velocity) using SMC-
DOB controller. (a) Overlaid centroid positions of the millirobot for both
conditions in comparison to the desired trajectory in the heart phantom. (b)
Tracking error over time for both cases.

Figure 7(a) shows the overlaid trajectories, and Fig. 7(b)
shows the corresponding tracking errors. A localized recirculation zone betweeny= 10mm and−5mm (Fig. 2(a)(ii))
introduced stronger disturbances at lower viscosity. In 4.3 cP
fluid, the millirobot exhibited noticeable oscillations within
this region, reflected by the wider spread of centroid positions
in Fig. 7(a). Outside this zone, the trajectory remained closely
aligned with the desired path.
Across three trials, the controller achieved an RMSE of
1.60±0.08mm, P95 error below 4 mm, and maximum error

y=10

Across three trials, the controller achieved an RMSE of
1.60±0.08mm, P95 error below 4 mm, and maximum error
of5.64±0.32mm in 4.3 cP fluid, compared to an RMSE of
1.00±0.02mm and maximum error of4.36±0.21mm in
20 cP fluid. These results demonstrate robust tracking despite
viscosity changes and the associated flow disturbances.

1.60{\pm}0.08

4.36\!\pm\!0.21

5.64{\pm}0.32

1.00\!\pm0.02

(a)

(b)

Fig. 7. Tracking performance of the SMC-DOB controller in 20 cP and
4.3 cP fluids under moderate pulsatile flow (7 cm/s peak speed). (a) Overlaid
centroid trajectories showing increased oscillations in the low-viscosity case,
particularly within the recirculation zone between inlet-1 and inlet-2. (b)
Comparison of RMSE, P95, and maximum errors across three trials for
both viscosities.

This work demonstrates autonomous navigation of a magnetic millirobot in a heart phantom under physiologically
relevant pulsatile flow. While electromagnetic actuation and
visual servoing have been widely studied, prior experimental validation has largely been limited to static or lowdisturbance environments. In contrast, pulsatile cardiac flow
introduces time-varying disturbances that degrade conventional controllers. The proposed SMC-DOB improves robustness by compensating for modeling uncertainty and transient
disturbances, enabling stable tracking under conditions where
PID and MPC show degraded performance.
Drag forces were estimated from steady-state CFD sim-

---

This study has several limitations. Experiments were conducted in a simplified phantom with moderate flow velocities
and static path planning. Future work will investigate higherflow conditions, more complex geometries, integration of
clinical imaging, and adaptive planning. Overall, these results
demonstrate the feasibility of robust autonomous magnetic
millirobot control under pulsatile flow disturbances.

References
[22]I. T. Gabe, J. H. Gault, J. Ross Jr, D. T. Mason, C. J. Mills, J. P.
[1]V. Iacovacci, E. Diller, D. Ahmed, and A. Menciassi, “Medical
microrobots,”Annu. Rev. Biomed. Eng., vol. 26, no. 1, pp. 561–591,
2024.
[2]M. Sitti, H. Ceylan, W. Hu, J. Giltinan, M. Turan, S. Yim, and E. Diller, [23]E.
“Biomedical applications of untethered mobile milli/microrobots,”
Proc. IEEE, vol. 103, no. 2, pp. 205–224, 2015.
[3]Y. Alapan, U. Bozuyuk, P. Erkoc, A. C. Karacakol, and M. Sitti,
“Multifunctional surface microrollers for targeted cargo delivery in [24]X. Chen, A. Bhattacharjee, L. O. Mair, S. Raval, P. Addepalli, O. Erin,
physiological blood flow,”Sci. Rob., vol. 5, no. 42, p. eaba5726, 2020.
[4]D. Wu, X. Wang, R. Li, C. Wang, Z. Ren, D. Pan, P. Ren, Y. Hu,
C. Xin, and L. Zhang, “Femtosecond laser–assisted printing of hard
magnetic microrobots for swimming upstream in subcentimeter-per-[25]M. Salehizadeh and E. Diller, “Two-agent formation control of magsecond blood flow,”Sci. Adv., vol. 11, no. 27, p. eadw1272, 2025.
[5]X. Du, Y. Wang, J. Law, K. Fang, H. Chen, Y. Liu, and J. Yu, “Active
exploration and reconstruction of vascular networks using microrobot [26]R. K. Shah and A. L. London,Laminar flow forced convection in ducts:
swarms,”Nat. Mach. Intell., vol. 7, no. 4, pp. 553–564, 2025.
[6]L. M. Baddour, W. R. Wilson, A. S. Bayer, V. G. Fowler Jr, I. M.
Tleyjeh, M. J. Rybak, B. Barsic, P. B. Lockhart, M. H. Gewitz, M. E. [27]O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional net-
Levisonet al., “Infective endocarditis in adults: diagnosis, antimicrobial therapy, and management of complications: a scientific statement
for healthcare professionals from the american heart association,”
Circulation, vol. 132, no. 15, pp. 1435–1486, 2015.
[7]E. Doutel, F. J. Galindo-Rosales, and L. Campo-Deano, “Hemodynam-˜
[28]W. Pryor, Y. Barnoy, S. Raval, X. Liu, L. Mair, D. Lerner, O. Erin,
ics challenges for the navigation of medical microbots for the treatment
of cvds,”Materials, vol. 14, no. 23, p. 7402, 2021.
[8]Z. Qi, M. Cai, B. Hao, Y. Cao, L. Su, X. Liu, K. F. Chan, C. Yang, and
L. Zhang, “Robust 3-d path following control framework for magnetic
helical millirobots subject to fluid flow and input saturation,”IEEE [29]N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr,
Trans. Cybern., 2024.
[9]J. Lu, Y. Liu, W. Huang, K. Bi, Y. Zhu, and Q. Fan, “Robust control
strategy of gradient magnetic drive for microrobots based on extended [30]Y. Li, Q. Zheng, and L. Yang, “Design of robust sliding mode control
state observer,”Cyborg Bionic Syst., 2022.
[10]H.-S. Lee, Y. Ko, J. Back, K.-S. Hong, and C.-S. Kim, “A doubleloop robust position and speed control of an electromagnetic actuator [31]E. Abdelaal, G. Plourde, J. MacHaalany, J. Arsenault, G. Rimac, J.-
against nonlinear model uncertainties,”Control Eng. Pract., vol. 160,
p. 106318, 2025.
[11]S. A. Abbasi, A. Ahmed, S. Noh, N. L. Gharamaleki, S. Kim,
A. M. B. Chowdhury, J.-y. Kim, S. Pane, B. J. Nelson, and H. Choi, ´
“Autonomous 3d positional control of a magnetic microrobot using
reinforcement learning,”Nat. Mach. Intell., vol. 6, no. 1, pp. 92–105, [32]C. Mitchell, P. S. Rahko, L. A. Blauwet, B. Canaday, J. A. Finstuen,
2024.
[12]Y. Yang, M. A. Bevan, and B. Li, “Hierarchical planning with deep
reinforcement learning for 3d navigation of microrobots in blood
vessels,”Adv. Intell. Syst., vol. 4, no. 11, p. 2200168, 2022.
[13]J. Zhou, M. Li, N. Li, H. Zhou, X. Wang, J. Wang, L. Liu, and N. Jiao,
“Model predictive control and replanning for magnetic microrobots
in fluid environments,”Sci. China Technol. Sci., vol. 68, no. 5, p.
1520302, 2025.
[14]Y. Liu, H. Wang, and Q. Fan, “Adaptive learning and sliding mode
control for a magnetic microrobot precision tracking with uncertainties,”IEEE Rob. Autom. Lett., vol. 8, no. 11, pp. 7767–7774, 2023.
[15]H. Wang, Y. Qiu, Y. Hou, Q. Shi, H.-W. Huang, Q. Huang, and
T. Fukuda, “Deep reinforcement learning-based collision-free navigation for magnetic helical microrobots in dynamic environments,”IEEE
Trans. Autom. Sci. Eng., vol. 22, pp. 7810–7820, 2024.
[16]M. Cai, Q. Wang, Z. Qi, D. Jin, X. Wu, T. Xu, and L. Zhang, “Deep
reinforcement learning framework-based flow rate rejection control of
soft magnetic miniature robots,”IEEE Trans. Cybern., vol. 53, no. 12,
pp. 7699–7711, 2022.

“Magnetically driven microrobots moving in a flow: a review,”Front.
Inf. Technol. Electron. Eng., vol. 24, no. 11, pp. 1520–1540, 2023.
[18]J. Jiang, Z. Yang, A. Ferreira, and L. Zhang, “Control and autonomy
of microrobots: Recent progress and perspective,”Adv. Intell. Syst.,
vol. 4, no. 5, p. 2100279, 2022.
[19]O. Erin, X. Chen, A. Bell, S. Raval, T. Schwehr, X. Liu, P. Addepalli,
L. O. Mair, I. N. Weinberg, Y. Diaz-Mercadoet al., “Strong magnetic
actuation system with enhanced field articulation through stacks of
individually addressed coils,”Sci. Rep., vol. 14, no. 1, p. 23123, 2024.
[20]A. Tsipis and E. Petropoulou, “Echocardiography in the evaluation of
the right heart,”US Cardiology Review, vol. 16, p. e08, 2022.
[21]A. Tejo-Otero, F. Fenollosa-Artes, ´ I. Achaerandio, S. Rey-Vinolas,
I. Buj-Corral, M. A.´ Mateos-Timoneda, and E. Engel, “Soft-tissuemimicking using hydrogels for the development of phantoms,”Gels,
vol. 8, no. 1, p. 40, 2022.
[22]I. T. Gabe, J. H. Gault, J. Ross Jr, D. T. Mason, C. J. Mills, J. P.
Schillingford, and E. Braunwald, “Measurement of instantaneous blood
flow velocity and pressure in conscious man with a catheter-tip velocity
probe,”Circulation, vol. 40, no. 5, pp. 603–614, 1969.
[23]E. D. Taco-Vasquez, F. Barrera, M. Serrano-Duenas, E. Jimenez,
A. Rocuts, and E. R. Perez, “Association between blood viscosity and
cardiovascular risk factors in patients with arterial hypertension in a
high altitude setting,”Cureus, vol. 11, no. 1, 2019.
[24]X. Chen, A. Bhattacharjee, L. O. Mair, S. Raval, P. Addepalli, O. Erin,
A. Bell, Y. Diaz-Mercado, and A. Krieger, “Mitigating singularities
in control of magnetic capsule endoscopes using a novel nested
electromagnetic coil system,”J. Med. Rob. Res., 2024.
[25]M. Salehizadeh and E. Diller, “Two-agent formation control of magnetic microrobots in two dimensions,”J. Micro-Bio Rob., vol. 12, pp.
9–19, 2017.
[26]R. K. Shah and A. L. London,Laminar flow forced convection in ducts:
a source book for compact heat exchanger analytical data. Academic
press, 2014.
[27]O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” inMedical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings,
part III 18. Springer, 2015, pp. 234–241.
[28]W. Pryor, Y. Barnoy, S. Raval, X. Liu, L. Mair, D. Lerner, O. Erin,
G. D. Hager, Y. Diaz-Mercado, and A. Krieger, “Localization and
control of magnetic suture needles in cluttered surgical site with blood
and tissue,” in2021 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), 2021, pp. 524–531.
[29]N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr,
R. Radle, C. Rolland, L. Gustafson ¨ et al., “Sam 2: Segment anything
in images and videos,”arXiv preprint arXiv:2408.00714, 2024.
[30]Y. Li, Q. Zheng, and L. Yang, “Design of robust sliding mode control
with disturbance observer for multi-axis coordinated traveling system,”
Comput. Math. Appl., vol. 64, no. 5, pp. 759–765, 2012.
[31]E. Abdelaal, G. Plourde, J. MacHaalany, J. Arsenault, G. Rimac, J.-
P. Dery, G. Barbeau, E. Larose, R. De Larochelli ´ ere, C. M. Nguyen `
et al., “Effectiveness of low rate fluoroscopy at reducing operator and
patient radiation dose during transradial coronary angiography and
interventions,”JACC: Cardiovasc. Interventions, vol. 7, no. 5, pp. 567–
574, 2014.
[32]C. Mitchell, P. S. Rahko, L. A. Blauwet, B. Canaday, J. A. Finstuen,
M. C. Foster, K. Horton, K. O. Ogunyankin, R. A. Palma, and E. J.
Velazquez, “Guidelines for performing a comprehensive transthoracic
echocardiographic examination in adults: recommendations from the
american society of echocardiography,”J Am Soc Echocardiogr,
vol. 32, no. 1, pp. 1–64, 2019.

[17]J. Miao, X. Wang, Y. Zhou, M. Ye, H. Zhao, R. Xu, and H. Qian,
“Magnetically driven microrobots moving in a flow: a review,”Front.
Inf. Technol. Electron. Eng., vol. 24, no. 11, pp. 1520–1540, 2023.
[18]J. Jiang, Z. Yang, A. Ferreira, and L. Zhang, “Control and autonomy

R e p.