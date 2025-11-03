# Optimization in Deep Networks

## 1) Learning Rate as an Implicit Geometry

The update $\theta_{t+1}=\theta_t-\eta, g_t$ is a discrete-time integrator of a stochastic differential equation. The scalar $\eta$ defines an **implicit metric** on parameter space: it rescales all directions uniformly, which is mismatched to the anisotropic curvature of deep models.

* **Edge of stability.** Training is fastest near the largest stable $\eta$ allowed by the sharpest local curvature (top Hessian eigenvalue $\lambda_{\max}$). In quadratic regions, stability requires $\eta < 2/\lambda_{\max}$; modern nets often operate near this boundary. Increasing $\eta$ trades bias for speed until oscillation sets in.
* **Loss–curvature coupling.** With BatchNorm and scale-invariant layers, rescaling weights can be absorbed by normalization; the “effective” learning rate becomes $\eta/|\theta|^2$ for certain directions. This explains the empirical benefit of **decoupled weight decay** and the robustness of large $\eta$ in normalized architectures.
* **Noise–curvature interaction.** In flat directions ($\lambda \ll \lambda_{\max}$), high $\eta$ accelerates progress without instability; in sharp directions it induces oscillation. Learning-rate sweeps reveal exactly where these regimes meet.

**If $\eta$ is too small:** optimization remains in a diffusion-dominated regime (noise $\gg$ drift), producing slow but stable descent with little curvature information used.
**If $\eta$ is too large:** the integrator overshoots along high-curvature eigenvectors; oscillation appears first in the components aligned with the top eigenspace.

### Table: Learning-rate sweeps (1e-4 → 1e-1)

| LR regime    | qualitative loss behavior         | spectral view (Hessian)                     | stochastic view                                | long-run bias                                           |
| ------------ | --------------------------------- | ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| **Too low**  | slow, near-flat decay             | steps ≪ 1/λ_max; under-integrates all modes | noise dominates drift                          | drifts within initial basin; weak curvature exploration |
| **Good**     | smooth or mildly noisy descent    | respects λ_max, advances on small λ_i       | beneficial noise aids escape to flatter basins | efficient exploration; stable convergence               |
| **Too high** | oscillation → spikes → divergence | violates stability along top eigenspace     | gradient noise amplified by underdamping       | chaotic trajectories; failure to settle                 |

---

## 2) Momentum as Spectral Filtering

Classical momentum (Polyak) introduces a velocity $v_t=\beta v_{t-1}+g_t$, update $\theta_{t+1}=\theta_t-\eta, v_t$.

* **Transfer function view.** In the frequency domain, momentum is a **low-pass filter** on gradient noise and a **phase-lead compensator** along consistent directions. For a quadratic $L=\tfrac12 \theta^\top H \theta$, the eigencomponent dynamics are second-order linear systems whose damping ratio depends on $(\eta,\beta,\lambda_i)$.
* **Ravine traversal.** In ill-conditioned valleys ($\lambda_{\parallel}\ll\lambda_{\perp}$), momentum suppresses zig-zagging by attenuating the high-frequency bouncing induced by steep directions, while accumulating drift along flat ones. Net effect: larger stable step size without sacrificing convergence.
* **Critical damping.** There exists a $\beta^*(\eta,\lambda_i)$ yielding critically damped trajectories per eigendirection. Because $\lambda_i$ varies across spectrum, any fixed $\beta$ is a compromise; the most visible benefit is reduced oscillation along the top eigenmodes.

**If momentum is absent:** progress is bottlenecked by the steepest direction; SGD “seesaws.”
**If momentum is excessive relative to $\eta$:** the system becomes underdamped; loss curves ring with period tied to $\lambda_{\max}$.

### Table: Momentum sweeps ($\beta$: 0 → 0.99)

| β regime  | trajectory shape             | frequency-domain view                  | curvature coupling         | risk profile               |
| --------- | ---------------------------- | -------------------------------------- | -------------------------- | -------------------------- |
| **0.0**   | zig-zag in ravines           | little low-pass filtering              | steep directions dominate  | inefficient progress       |
| **~0.9**  | smooth arcs, quick alignment | attenuates high-freq noise, phase lead | expands stable LR window   | robust “sweet spot”        |
| **~0.99** | long spirals, slow damping   | underdamped near λ_max                 | sensitive to schedule & LR | ringing / overshoot likely |

---

## 3) SGD vs. SGD+Momentum: Different Biases

* **SGD** (no momentum) estimates the gradient with mini-batch noise; its stochasticity helps exit narrow basins but wastes steps in anisotropic regions.
* **SGD+M** introduces **temporal correlation** in the update direction, effectively **reducing gradient variance** and **increasing step coherence**. The bias is toward motion that integrates information over time, which empirically accelerates convergence at fixed generalization.

The generalization gap is often explained not by asymptotic minima identity, but by the **noise scale** each induces during training (see §8): momentum reduces effective noise for the same batch size and $\eta$, which can move the solution toward sharper regions unless counterbalanced by schedules or decay.

### Table: SGD vs. SGD+Momentum

| optimizer | geometry of steps                                        | variance properties                       | typical convergence               | generalization tendency                                             |
| --------- | -------------------------------------------------------- | ----------------------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| **SGD**   | follows instantaneous gradient; jagged in anisotropy     | higher per-step variance at fixed batch   | slower in ill-conditioned valleys | often flatter minima when heavily scheduled                         |
| **SGD+M** | integrates direction over time; straighter valley travel | reduced effective noise for same LR/batch | faster wall-clock descent         | can bias toward sharper regions if schedules/decay aren’t balancing |

---

## 4) What Loss Curves Tell About Dynamics

* **Smooth monotone decrease** → drift dominates noise; updates are well inside the stability region for top curvature.
* **Noisy but trending down** → noise comparable to drift; often correlated with smaller batches or stronger augmentation, frequently yielding flatter minima later.
* **Oscillatory envelopes** → underdamped dynamics along top eigenmodes; either $\eta$ or $\beta$ is too high relative to $\lambda_{\max}$.
* **Early plateau** → either $\eta$ below curvature-aware scale (under-integration), poor signal propagation from initialization, or a bottleneck layer limiting gradient flow.

The **alignment** between train/validation curves indicates whether noise acts as regularization (benign) or the model is entering sharp regions (train ↓, val ↑ with rising gradient norms).

### Table: Loss-curve morphology (diagnostic)

| observed curve       | dominant regime                       | spectral implication                 | stochastic implication                  |
| -------------------- | ------------------------------------- | ------------------------------------ | --------------------------------------- |
| smooth monotone ↓    | drift ≫ noise                         | LR inside stability across top modes | batch/aug noise subdued                 |
| noisy but trending ↓ | mixed regime                          | stable, but small modes still noisy  | exploration aided by noise              |
| periodic envelopes   | underdamped top modes                 | LR/β near edge for λ_max             | noise resonantly amplified              |
| early plateau        | under-integration or poor signal flow | LR below useful scale for small λ_i  | diffusion dominates; weak curvature use |

---

## 5) Weight Trajectories: Geometry Made Visible

Projecting checkpoints onto the top PCs (of parameters, features, or logits) shows **shape of motion**:

* **Zig-zags** (SGD, high curvature mismatch): motion alternates orthogonal to valley floor.
* **Arcs/spirals** (momentum near edge): underdamped oscillations around the valley floor; amplitude decays with damping ratio.
* **Elliptical, axis-aware steps** (adaptive methods): step lengths shrink along historically volatile axes (estimated by second-moment stats), approximating a diagonal preconditioner.

The geometry is **spectral**: trajectories align with the leading eigenspaces of the empirical Hessian/Fisher; changes in optimizer or hyperparameters mostly modify **how fast each eigencomponent decays**.

### Table: Weight-trajectory geometry (2D toy or PCA)

| pattern                  | typical cause                    | spectral readout                     | optimizer signature   |
| ------------------------ | -------------------------------- | ------------------------------------ | --------------------- |
| sharp zig-zags           | anisotropy with no momentum      | alternating across steep/flat axes   | SGD (β=0)             |
| smooth arcs              | temporal coherence from momentum | gradual alignment to valley floor    | SGD+M, Adam (β₁>0)    |
| outward spirals          | LR/β beyond damping              | energy pumped into top eigenmodes    | any method near edge  |
| ellipses aligned to axes | per-coordinate scaling           | diagonal preconditioning of spectrum | RMSProp / Adam family |

---

## 6) Convergence Stability as a Spectral Constraint

For quadratic $H\succeq 0$, stability reduces to per-eigenvalue conditions. With momentum, the **characteristic polynomial** yields a damping ratio $\zeta(\eta,\beta,\lambda_i)$. Convergence rate is set by the **slowest decaying mode** (often near $\lambda_{\min}^+$), while the **stability limit** is set by $\lambda_{\max}$.

Thus any scalar optimizer faces a dilemma: choose $\eta$ to respect $\lambda_{\max}$ while not starving progress along small $\lambda_i$. Schedules (cosine/one-cycle) navigate this by **starting conservative** (to avoid blow-up) then **expanding the attraction basin** before annealing to reduce noise and settle.

### Table: Convergence stability regimes

| regime                    | mathematical constraint       | phenomenology             | asymptotic effect                |
| ------------------------- | ----------------------------- | ------------------------- | -------------------------------- |
| **stable, overdamped**    | η·λ_max ≪ 1                   | slow but steady           | safe convergence, limited speed  |
| **stable, near-critical** | η ≲ 2/λ_max (with β tuned)    | fastest practical descent | preferred operating point        |
| **underdamped**           | η, β too aggressive for λ_max | ringing, spikes           | fragile; sensitive to data order |
| **unstable**              | η ≫ 2/λ_max                   | divergence / NaNs         | training collapse                |

---

## 7) Initialization as Distribution Shaping

Initialization determines early-time statistics of activations and gradients:

* **Xavier (Glorot)** preserves variance across layers for symmetric activations; **He (Kaiming)** corrects for ReLU half-space sparsity.
* Proper scaling maintains the **Jacobian’s singular values** around 1, preventing exponential growth/decay of signals (the “dynamical isometry” idea).
* With BatchNorm/LayerNorm, the network acquires **scale invariances** that couple with optimizer choices: e.g., decoupled decay acts on weight norms even when the loss surface is invariant to rescaling, correcting the implicit bias of adaptive methods.

Bad initialization corrupts the local quadratic model: “gradient” directions become dominated by saturation or exploding channels; the optimizer’s apparent instability is often **an artifact of ill-posed local curvature** created at $t=0$.

### Table: Initialization (distributional role)

| scheme            | activation class | preserved statistic           | early-time gradient flow    | typical effect                      |
| ----------------- | ---------------- | ----------------------------- | --------------------------- | ----------------------------------- |
| **Xavier/Glorot** | tanh/sigmoid     | var(in) ≈ var(out)            | avoids saturation           | stable but slower with ReLU         |
| **He/Kaiming**    | ReLU/variants    | accounts for half-space zeros | maintains signal norm       | faster early descent                |
| **Orthogonal**    | RNN/long depth   | singular-values near 1        | resists vanishing/exploding | improved temporal credit assignment |
| **Naïve uniform** | any              | none                          | saturates or explodes       | misleading LR/β behavior            |

---

## 8) Batch Size as Noise Control

Stochastic gradients can be decomposed as $g_t = \nabla L(\theta_t) + \xi_t$ with $\mathbb{E}[\xi_t]=0$, $\mathrm{Var}(\xi_t)\propto 1/B$.

* **Noise scale** $S \sim \eta/B$ (modulated by loss curvature and augmentation) influences which basins are reachable. Higher $S$ aids **escapes from sharp minima** and can bias toward **flatter** regions; lower $S$ yields precise but potentially **sharper** solutions.
* **Large-batch regimes** reduce $S$; to preserve similar stochastic dynamics, one often increases $\eta$ (up to stability), injects noise (regularization/augmentation), or adopts flatness-seeking updates (e.g., SAM).

Thus, batch size is not just throughput; it **shifts the stochastic thermodynamics** of training.

### Table: Batch size as noise control

| batch regime | noise scale (∝ η/B) | landscape bias                         | curvature sampling                  | generalization tendency                  |
| ------------ | ------------------- | -------------------------------------- | ----------------------------------- | ---------------------------------------- |
| **small**    | high                | favors wider basins                    | stochastic escape from sharp minima | often better                             |
| **medium**   | moderate            | balanced                               | decent exploration & signal         | strong practical zone                    |
| **large**    | low                 | favors sharp minima unless compensated | precise but less exploratory        | can degrade without extra regularization |

---

## 9) RMSProp, Adam, and AdamW: Diagonal Preconditioning, State, and Decay

All three implement **per-coordinate scaling** approximating a diagonal inverse-curvature model.

* **RMSProp.** Tracks $v_t=\rho v_{t-1}+(1-\rho)g_t^2$; steps $g_t/\sqrt{v_t+\epsilon}$. It damps motion along historically high-variance directions, behaving like a **time-varying diagonal preconditioner**. Converges quickly in ravines but may stall on plateaus lacking gradient-magnitude contrast.
* **Adam.** Adds first-moment $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ with bias correction: $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$. Update $\Delta\theta\propto \hat m_t/(\sqrt{\hat v_t}+\epsilon)$. This couples momentum and adaptivity: the step direction is **history-smoothed**, and its **length** is normalized by past volatility. Adam’s stability at larger $\eta$ stems from suppressing steps where gradients are volatile, effectively flattening the spectrum.
* **AdamW (decoupled weight decay).** Classical L2 modifies the gradient $g\leftarrow g+\lambda\theta$, which interacts with adaptivity and the data-dependent preconditioner. AdamW **separates** decay as $\theta\leftarrow(1-\eta\lambda)\theta$, aligning with scale-invariant layers and preserving the intended regularization effect. Empirically this **restores the link** between weight norms and generalization, particularly under BatchNorm.

**Trade-offs.** Adaptive methods rapidly fit idiosyncratic modes (fast training) but can bias toward **sharper** minima; SGD+M often yields **flatter** solutions when tuned. The difference is less about destination and more about **the path distribution** through parameter space, modulated by $(\eta, B)$ and decay.

### Table: Optimizers — taxonomy of behaviors

| optimizer   | update structure                    | implicit metric                          | strengths                 | typical caveats                      |
| ----------- | ----------------------------------- | ---------------------------------------- | ------------------------- | ------------------------------------ |
| **SGD**     | −η·g                                | Euclidean scalar                         | simple, reliable          | needs tight LR control               |
| **SGD+M**   | −η·EMA(g)                           | temporal smoothing                       | faster in anisotropy      | underdamping risk                    |
| **RMSProp** | −η·g/√EMA(g²)                       | diagonal preconditioner (variance-aware) | ravine traversal          | plateaus if variance cues vanish     |
| **Adam**    | −η·EMA(g)/√EMA(g²) (bias-corrected) | temporal + diagonal                      | rapid early fit; robust   | sharper minima bias if unconstrained |
| **AdamW**   | Adam + decoupled decay              | as above + norm control                  | state-of-practice default | decay must be meaningfully set       |

---

## 10) Weight Decay, Flatness, and Sharpness

* **Sharpness** can be probed by $\lambda_{\max}(H)$ or by local loss increase under small perturbations. Decay penalizes large norms, indirectly limiting the model’s capacity to inhabit sharp basins where small parameter changes cause large loss changes.
* **Coupled L2 vs decoupled decay.** With adaptive preconditioners, coupling L2 into $g$ causes per-coordinate rescaling of the regularizer, **distorting** its effect; decoupling ensures a geometrically coherent contraction in parameter norm, independent of the data-dependent preconditioner.
* **Late-phase dynamics.** As learning rate anneals, stochastic noise decreases; without adequate decay or augmentation, the optimizer “settles” into sharper regions. Decay counters this drift by biasing the basin choice toward flatter neighborhoods.

### Table: Weight decay & sharpness (coupled vs decoupled)

| regularizer style     | mechanism          | interaction with adaptivity               | geometric effect               | empirical tendency                |
| --------------------- | ------------------ | ----------------------------------------- | ------------------------------ | --------------------------------- |
| **L2 (coupled)**      | add λθ to gradient | rescaled per-coordinate by preconditioner | distorts intended norm penalty | inconsistent with adaptive stats  |
| **Decoupled (AdamW)** | direct θ ← (1−ηλ)θ | independent of g-scaling                  | coherent norm contraction      | better stability & generalization |

---

## 11) Interpreting the Canonical Experiments

**Learning-rate sweeps (1e−4→1e−1).** Reveal the stability frontier governed by $\lambda_{\max}$. The usable window is wide with adaptivity (suppressed effective curvature) and narrower with SGD in unnormalized nets.

**Momentum sweeps (0→0.99).** Trace the damping ratio across the spectrum. Peak efficiency occurs where ringing is barely avoided; beyond that, the top eigenspaces exhibit periodic energy exchange (loss “breathing”).

**SGD vs SGD+Momentum.** The difference appears strongest in anisotropic toy problems or early training on real nets: momentum straightens trajectories and advances faster along flat manifolds of near-symmetry (e.g., weight permutations).

**Loss curves.** Oscillation period often correlates with $\sqrt{\beta}/\eta$ and the top eigenvalue; abrupt scheduler changes excite these modes (visible spikes).

**Weight trajectories (2D/PCA).** Spirals mark underdamped top modes; ellipses with aligned axes signal effective diagonal preconditioning; jagged paths indicate curvature mismatch.

**Convergence stability.** A system can be stable on average but unstable conditionally: augmentations or batch composition change the instantaneous spectrum, intermittently breaching the edge of stability and causing rare spikes.

**Initializations.** He vs Xavier differences manifest in early-time gradient norms and activation sparsity; improper scales push the system either into diffusion (tiny gradients) or chaos (exploding activations), confounding interpretation of LR and momentum.

**Batch size variations.** Holding $\eta$ fixed while increasing $B$ reduces noise scale $S$, typically improving fit speed but **narrowing** the implicit regularization provided by stochasticity; test sharpness increases unless decay/regularization adjusts.

**RMSProp/Adam.** Adaptivity reshapes the effective metric: steps become **Mahalanobis-like** with covariance approximated diagonally. Fast early progress is the signature; late-time behavior depends on decay and schedules to avoid sharp minima bias.

### Table: Canonical experiment → core takeaway

| experiment              | the invariant it reveals                                |
| ----------------------- | ------------------------------------------------------- |
| **LR sweep**            | stability boundary set by top curvature; usable LR band |
| **Momentum sweep**      | damping ratio across spectrum; coherence vs ringing     |
| **SGD vs SGD+M**        | benefit of temporal correlation in anisotropy           |
| **Loss curves**         | relative strength of drift vs noise; schedule resonance |
| **Weight trajectories** | geometric structure (zig-zag vs ellipse vs spiral)      |
| **Init variants**       | early signal propagation; local quadratic fidelity      |
| **Batch size sweep**    | noise-controlled basin selection                        |
| **Optimizer bench**     | metric (preconditioning) vs path-dependent minima       |

---

## 12) Failure Modes as Mismatches

* **Oscillation:** $(\eta,\beta)$ incompatible with $\lambda_{\max}$; spectral underdamping.
* **Divergence:** nonlinearities drive you outside the quadratic basin; activations/BN statistics shift; step clips are required or $\eta$ is far too high.
* **Plateaus:** smallest nonzero eigenvalues dominate; scalar step cannot amplify these modes without violating stability upstream; preconditioning (adaptive methods) or schedules relieve the bottleneck.
* **Overfitting drift:** late-time low noise + insufficient decay biases toward sharp basins; decoupled decay or flatness-promoting updates counteract.

### Table: Signals → latent causes (interpretation map)

| signal                               | likely latent cause                          | lens to confirm                                                        |
| ------------------------------------ | -------------------------------------------- | ---------------------------------------------------------------------- |
| ringing with stable average ↓        | underdamped top eigenmodes                   | PCA trajectories show spirals; spectrum proxies high                   |
| sudden spikes post warmup            | LR jump vs current curvature                 | inspect LR schedule; track gradient norms                              |
| train ↓, val ↑ with rising sharpness | settling into sharp basin                    | perturbation sharpness / λ_max estimates increase                      |
| flat early loss across LRs           | poor initialization / saturated nonlinearity | activation histograms; Jacobian stats                                  |
| smooth but slow                      | overdamped / small η                         | compare to spectrum proxy; increase η until near-edge behavior appears |

---

## 13) Why Visualizations Matter

* **Loss curves** measure **temporal regularity** and stability.
* **Trajectory plots** expose **geometric mode structure** (damping, preconditioning).
* **Gradient/activation norms** reveal **signal propagation health** and proximity to nonlinear saturation.
* Together they separate **scalar miscalibration** (LR, momentum) from **structural issues** (init, normalization, optimizer geometry).

---

## 14) Synthesis

Deep optimization is an interplay between:

* **Curvature** (Hessian spectrum),
* **Noise** (stochastic gradients shaped by batch and augmentation),
* **Geometry** (implicit metric from $\eta$, momentum, and preconditioning),
* **Invariances** (normalization layers and scale symmetries),
* **Regularization** (decay shaping norm and sharpness).

Learning rate sets the **energy** of exploration; momentum sets the **temporal coherence**; batch size controls the **thermodynamic noise**; initialization determines whether information **flows**; adaptive methods reshape the **metric**; decoupled decay anchors the **norm**. The best behavior emerges near—but not beyond—the edge of stability, where curvature is respected, noise is productive, and geometry is aligned with the model’s invariances.

---
