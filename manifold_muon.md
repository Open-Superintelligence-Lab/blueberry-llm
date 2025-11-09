THINKING MACHINES
Tinker
Blog
Join us
Modular Manifolds
Jeremy Bernstein
Sep 26, 2025
77 777777 777777777777                   777777777777777777                7777777777777777777777777            777777777777777777777777777777          77777777777777777777777777777777777777      77777777777777777777777777777777777777777777   177777777777777777777777777777777777777777777772 445377777777777777777777777777777777777777775800 444445377777777777777777777777777777777715000000 444444445277777777777777777777777777716800000000 444444444444217777777777777777777736000000000000 444444444444444517777777777777739000000000000000 444444444444444444517777777728000000000000000000 444444444444444444444537748000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 44444444444444444444460000000000000000000000   44444444444444444460000000000000000000      444444444444446000000000000000          44444444444460000000000000               44444444446000000000                    444446000000 446800 40
4444
77 777777 777777777777                   777777777777777777                7777777777777777777777777            777777777777777777777777777777          77777777777777777777777777777777777777      77777777777777777777777777777777777777777777   177777777777777777777777777777777777777777777772 445377777777777777777777777777777777777777775800 444445377777777777777777777777777777777715000000 444444445277777777777777777777777777716800000000 444444444444217777777777777777777736000000000000 444444444444444517777777777777739000000000000000 444444444444444444517777777728000000000000000000 444444444444444444444537748000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 44444444444444444444460000000000000000000000   44444444444444444460000000000000000000      444444444444446000000000000000          44444444444460000000000000               44444444446000000000                    444446000000 446800 40
4444
77 777777 777777777777                   777777777777777777                7777777777777777777777777            777777777777777777777777777777          77777777777777777777777777777777777777      77777777777777777777777777777777777777777777   177777777777777777777777777777777777777777777772 445377777777777777777777777777777777777777775800 444445377777777777777777777777777777777715000000 444444445277777777777777777777777777716800000000 444444444444217777777777777777777736000000000000 444444444444444517777777777777739000000000000000 444444444444444444517777777728000000000000000000 444444444444444444444537748000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 444444444444444444444446000000000000000000000000 44444444444444444444460000000000000000000000   44444444444444444460000000000000000000      444444444444446000000000000000          44444444444460000000000000               44444444446000000000                    444446000000 446800 40
4444
Introduction
The shape of a manifold optimizer
Manifold Muon
Modular manifolds
Directions for future work
Further reading
Citation
When we train large neural networks, we need to keep them healthy. We do not want the tensors in the network—either the weights, activations or gradients—to grow too large or too small. Very small and very large tensors cause a variety of problems not just limited to numerical underflow and overflow. For example, weight matrices changing size during training makes it harder to design training algorithms—since the relative size of updates to weights has a significant impact on the speed of learning.

The gold standard for keeping tensors healthy is to normalize them. Normalization is commonplace for activation vectors, where we use techniques like layer norm to put the activations on a good scale before passing them to the next layer. It is also commonplace to normalize gradient updates, where we can interpret fast training algorithms like the Muon optimizer as spectrally normalizing the updates. Normalization provides us with certainty about the sizes of tensors—without needing to check Wandb!—and when training large neural networks with many interacting components, having certainty about the network internals is valuable.

Normalization is less commonly applied to weight matrices, although it is not unheard of. For example, the EDM2 diffusion model codebase uses weight constraints and the authors report benefits in their paper. And BiT uses weight standardization. Various other techniques have been proposed but are not common practice in modern large-scale training.For some more examples, see Salimans et al, 2016, Miyato et al, 2018 and our paper Liu et al, 2021. Normalizing the weight matrices might be a good idea for a few reasons. Weight constraints make understanding the relative size of optimization updates easier. They remove the problem of weight norms exploding. They allow us to focus hyperparameter tuning effort on tensors whose size matters most. They can force matrices to have a small condition number, making their behaviour more predictable. And relatedly, weight constraints facilitate Lipschitz guarantees for robustness to perturbations.

This post covers one appealing way to constrain the weight matrices of a neural network—by keeping the tensors constrained to submanifolds at each layer. This opens the door to re-thinking optimization, as we can co-design optimization algorithms with these manifold constraints. As an example, we proposeThis algorithm builds on work from Jianlin Su and Franz Louis Cesista, as discussed further below. a manifold version of the Muon optimizer whose weights are constrained to the Stiefel manifold: the manifold of matrices with unit condition number. We conclude the post by defining the idea of a modular manifold, which is a composable manifold that attempts to make it easier to scale up and train large networks.

Our goal in writing this post is to provide an introduction to a research area that we are excited about, and highlight many directions for future work. We would love to see more work from the community on the topics mentioned at the end of the post!

The shape of a manifold optimizer#
This section works through the simplest example of learning on a manifold: a vector parameter constrained to a hypersphere in 
R
d
R 
d
 . The vector parameter is trained to minimize a loss function defined over the full space 
R
d
R 
d
 . This setup might be useful for, say, individual embedding vectors in a transformer model. This section will be a good warmup for the following section on manifold Muon that considers matrix parameters.

We will not be too formal about the definition of a manifold here: it is enough to understand that a manifold is a curved surface that looks flat when you zoom in close enough. The locally flat approximation at a point on the manifold is called the tangent space to the manifold, as visualized in Figure 1:

The sphere in three dimensions—or the hypersphere in higher dimensions—is a manifold. The locally flat approximation at a point on the manifold is called the tangent space to the manifold and is visualized as the red plane in the figure.
We can characterize the hypersphere in 
d
d dimensions as the set of points 
w
∈
R
d
w∈R 
d
  of unit Euclidean norm. And the tangent space at a point 
w
w on the hypersphere is the set of all vectors 
a
∈
R
d
a∈R 
d
  that are orthogonal to 
w
w.

To keep the weights constrained to the manifold, we could use a non-manifold optimizer and just project the weights back to the manifold after each step. Instead, we are interested in designing methods that take steps in the tangent space. The reason is that we would like to be able to equate the learning rate of our optimizer with the actual length of the optimization step. But if the optimization steps are pointing significantly off manifold and then being projected back, this nice property does not hold. Similar motivation is given in Section 2.3 of the EDM2 paper.

Before we can design a training algorithm for this manifold, something important we need to decide on is how to measure distanceFor a manifold to be “Riemannian”, the distance measure must be induced by an inner product. The Euclidean (
ℓ
2
ℓ 
2
​
 ) norm is induced by an inner product, but the Manhattan (
ℓ
1
ℓ 
1
​
 ) distance is not. in the tangent space. A common choice is the Euclidean distance, but we could also choose to measure distance in other ways, as visualized in Figure 2. In the next section, we will talk about choosing a distance measure based on the functionality of the module.

Inscribing unit balls in the tangent space for different distance measures. The 
ℓ
2
ℓ 
2
​
  (Euclidean) unit ball is a circle while the 
ℓ
1
ℓ 
1
​
  (Manhattan) unit ball is a diamond.
Crucially, the choice of distance measure changes the direction of the best optimization step. If the distance measure is non-Euclidean, then for a fixed length step, we may be able to move further in the direction of the gradientBy gradient, we mean the partial derivative of the loss with respect to the weights. Mathematicians reserve the term gradient for something else in Riemannian geometry. by not following the gradient direction exactly! This concept is visualized in Figure 3.

How geometry influences the direction of the best optimization step. The pink arrow represents the raw gradient—meaning the partial derivative of the loss with respect to the weights. The yellow diamond denotes the 
ℓ
1
ℓ 
1
​
  unit ball. The green arrow is the unit vector pointing furthest in the direction of the gradient. Notice how the green arrow is not parallel to the pink arrow. (In practice, the pink arrow need not lie in the tangent space, although the green arrow will do by construction.) Try dragging the pink arrow to see how the best update direction changes.
To see how this works out in math, we can formulate the optimal update direction given a manifold constraint and a distance measure as itself solving a constrained optimization problem. We will demonstrate this for the case of the hypersphere equipped with the Euclidean norm. Letting 
g
g denote the gradient, 
w
w the current point on the hypersphere, 
a
a the update direction and 
η
η the learning rate, we need to solve:

min
⁡
a
∈
R
d
a
⊤
g
⏟
linear change in loss
such that
∥
a
∥
2
=
η
⏟
size constraint
and
a
⊤
w
=
0
⏟
tangent constraint
.
(
⋆
)
a∈R 
d
 
min
​
  
linear change in loss
a 
⊤
 g∥a∥ 
2
​
 =1
​
 
​
 such that 
size constraint
∥a∥ 
2
​
 =η
​
 
​
 and 
tangent constraint
a 
⊤
 w=0∥a∥ 
2
​
 =1
​
 
​
 .(⋆)
Mapping back to the visual language of Figures 1, 2 and 3, this formula says that the green arrow (optimal value of 
a
a) must belong to the red tangent hyperplane (
a
⊤
w
=
0
a 
⊤
 w=0) and must also lie on a yellow circle of radius 
η
η (
∥
a
∥
2
=
η
∥a∥ 
2
​
 =η). To solve 
(
⋆
)
(⋆), we can apply the method of Lagrange multipliers. The relevant Lagrangian function is given by:

L
(
a
,
λ
,
μ
)
=
a
⊤
g
+
λ
2
⋅
(
a
⊤
a
−
η
)
+
μ
⋅
(
a
⊤
w
)
,
L(a,λ,μ)=a 
⊤
 g+ 
2
λ
​
 ⋅(a 
⊤
 a−η)+μ⋅(a 
⊤
 w),
where 
λ
λ and 
μ
μ are Lagrange multipliers. Setting the derivative of the Lagrangian with respect to 
a
a to zero and applying the constraints to solve for 
λ
λ and 
μ
μ, the optimal update 
a
o
p
t
a 
opt
​
  ends up being given by the following formula:

a
o
p
t
=
−
η
×
g
−
w
w
⊤
g
∥
g
−
w
w
⊤
g
∥
2
.
a 
opt
​
 =−η× 
∥g−ww 
⊤
 g∥ 
2
​
 
g−ww 
⊤
 g
​
 .
In words, the optimal update is given by subtracting out the radial component from the gradient, normalizing and multiplying by the learning rate. Since this update lies in the tangent space, actually a very smallFor a learning rate 
η
η, the effect of the retraction map is 
O
(
η
2
)
O(η 
2
 ) small, so the learning rate almost equals the length of the step. correction is needed to stay on the manifold. The correction is known as a “retraction map” and is visualized in Figure 4:

Visualizing the retraction map. The green arrow is the update taken in the tangent space. Since for large step sizes the tangent space starts to diverge from the manifold, we need to project the updated weights back to the manifold using the retraction map—illustrated by the purple arrow.
We can solve for the retraction map by applying Pythagoras’ theorem to Figure 4. For a unit hypersphere and a step of length 
η
η, the hypotenuse has length 
1
+
η
2
1+η 
2
 
​
  and therefore the retraction map for the hypersphere equipped with the Euclidean norm is simply given by dividing the updated weights through by 
1
+
η
2
1+η 
2
 
​
 . Putting everything together, the full manifold optimization algorithm is then given by:

w
←
1
1
+
η
2
[
w
−
η
×
g
−
w
w
⊤
g
∥
g
−
w
w
⊤
g
∥
2
]
.
w← 
1+η 
2
 
​
 
1
​
 [w−η× 
∥g−ww 
⊤
 g∥ 
2
​
 
g−ww 
⊤
 g
​
 ].
As an exercise for the reader: try calculating the Euclidean norm of the updated weight vector and check that the updated weight vector indeed lies on the hypersphere.

To summarize this section, a first-order manifold optimizer has three steps:

Find the tangent vector of unit length that goes furthest in the gradient direction.
Multiply this direction by the learning rate and subtract from the weights;
Retract the updated weights back to the manifold.
There are two decisions to make in applying this procedure: what manifold constraint we should use and how we should measure length. By making different decisions, we can generate different optimization algorithms as shown in the following table.

Manifold	Norm	Optimizer
Euclidean 
R
n
R 
n
 	Euclidean norm	vanilla gradient descent
Euclidean 
R
n
R 
n
 	infinity norm	sign gradient descent
hypersphere 
S
n
S 
n
 	Euclidean norm	hyperspherical descent
matrix space 
R
m
×
n
R 
m×n
 	spectral norm	Muon
Stiefel manifold 
⊂
R
m
×
n
⊂R 
m×n
 	spectral norm	manifold Muon
We will derive the final algorithm in the table, manifold Muon, in the next section. To design a manifold constraint and a distance function for a matrix parameter, we shall think carefully about the role that a weight matrix plays inside a neural network.

Manifold Muon#
A typical weight matrix 
W
W in a transformer is a “vector-multiplier”, meaning that it transforms an input vector 
x
x into an output vector 
y
=
W
x
y=Wx. We will design a manifold constraint and a distance function so that the matrix acts in a good way on input vectors: the matrix should not produce excessively small or large outputs, and updates to the matrix should not cause the output vector to change too much or too little.

A good way to think about how a matrix acts on vectors is through the singular value decomposition, illustrated in Figure 5. The SVD decomposes a matrix in a way that tells us how the matrix stretches input vectors along different axes.

m
x
n
m
x
k
k
x
k
k  rank M
=
k
x
n
=
M
U
Σ
V
T
The singular value decomposition. A matrix 
M
∈
R
m
×
n
M∈R 
m×n
  of rank 
k
k can always be decomposed as 
M
=
U
Σ
V
⊤
M=UΣV 
⊤
 , where 
U
∈
R
m
×
k
U∈R 
m×k
  and 
V
∈
R
n
×
k
V∈R 
n×k
  have orthonormal columns and 
Σ
∈
R
k
×
k
Σ∈R 
k×k
  is a diagonal matrix with only positive entries. The entries of 
Σ
Σ are called the singular values of 
M
M. The singular values measure the stretching effect that the matrix has on vectors that align with the corresponding columns of 
U
U and 
V
V.
We would like the matrix to have a stretching effect close to one, so we will choose a matrix manifold where all the singular values are exactly one. This matrix manifold is known formally as the Stiefel manifold. We can assume without loss of generality that we are dealing with a tall matrix (
m
≥
n
m≥n), and then the Stiefel manifold can be equivalently defined as the following set:

S
t
i
e
f
e
l
(
m
,
n
)
:
=
{
W
∈
R
m
×
n
∣
W
T
W
=
I
n
}
.
Stiefel(m,n):={W∈R 
m×n
 ∣W 
T
 W=I 
n
​
 }.
Furthermore, one may show that a matrix 
A
∈
R
m
×
n
A∈R 
m×n
  lies tangentNotice that the Stiefel constraint 
W
T
W
=
I
n
W 
T
 W=I 
n
​
  directly generalizes the hyperspherical constraint 
w
⊤
w
=
1
w 
⊤
 w=1 from the previous section. Similarly, the tangent space condition generalizes the hyperspherical one that 
a
⊤
w
=
0
a 
⊤
 w=0. to the Stiefel manifold at matrix 
W
W if and only if:

A
⊤
W
+
W
⊤
A
=
0.
A 
⊤
 W+W 
⊤
 A=0.
To design a manifold optimizer for the Stiefel manifold, all that remains is to choose a distance function. To limit the maximum stretching effect the weight update can have on an input vector, we will choose the spectral norm, which measures the largest singular value of a matrix. Although this only limits the maximum effect the update can have, since the optimizer we derive will saturate this bound, it will turn out to prevent the minimum effect of the update from being too small.There are some exceptions to this statement, such as when a weight matrix has a fan-out less than its fan-in, in which case we cannot escape from the matrix and its updates having a null space and mapping some inputs to zero.

The idea of doing gradient descent under a spectral norm constraint is what led to the Muon optimizer and, when combined with the Stiefel manifold constraint, we obtain a problem that we shall call manifold Muon:

min
⁡
A
∈
R
m
×
n
trace
⁡
(
G
T
A
)
⏟
linear change in loss
such that
∥
A
∥
spectral
≤
η
⏟
size constraint
and
A
T
W
+
W
T
A
=
0
⏟
tangent constraint
.
(
†
)
A∈R 
m×n
 
min
​
  
linear change in loss
trace(G 
T
 A)
​
 
​
 such that 
size constraint
∥A∥ 
spectral
​
 ≤η
​
 
​
 and 
tangent constraint
A 
T
 W+W 
T
 A=0∥A∥ 
spectral
​
 =η
​
 
​
 .(†)
The manifold Muon problem 
(
†
)
(†) directly generalizes problem 
(
⋆
)
(⋆) from the previous section. Solving 
(
†
)
(†) is harder than solving 
(
⋆
)
(⋆), and here we will present a numerical solution inspiredI figured out how to solve manifold Muon in the square case late last year, but I was unable to solve the full rectangular case and thus posed the problem as an open problem on the Modula docs. Jianlin Su solved the problem this summer by taking a Lagrangian approach and working out a fixed point iteration on the optimality condition. I saw an early version of Jianlin’s work (which did not quite work yet) and also related work by Franz Louis Cesista, and I was able to work out the dual ascent algorithm presented here. by work done by Jianlin Su and Franz Louis Cesista.

Our key insight is that 
(
†
)
(†) is a convex optimization problem that may be solved via a standard method known as dual ascent. Here we will just sketch the main idea, but you can find a more detailed derivation on this page.

Similar to Jianlin’s approach, we introduce a matrix of Lagrange multipliers 
Λ
∈
R
n
×
n
Λ∈R 
n×n
 . We then apply a series of transformations to convert the problem 
(
†
)
(†) from a constrained minimization problem to an unconstrained maximization problem:

(
†
)
=
min
⁡
∥
A
∥
s
p
e
c
t
r
a
l
≤
η
max
⁡
Λ
trace
⁡
G
⊤
A
+
trace
⁡
Λ
⊤
(
A
⊤
W
+
W
⊤
A
)
=
min
⁡
∥
A
∥
s
p
e
c
t
r
a
l
≤
η
max
⁡
Λ
trace
⁡
A
⊤
(
G
+
2
W
(
Λ
+
Λ
⊤
)
)
=
max
⁡
Λ
min
⁡
∥
A
∥
s
p
e
c
t
r
a
l
≤
η
trace
⁡
A
⊤
(
G
+
2
W
(
Λ
+
Λ
⊤
)
)
=
max
⁡
Λ
−
η
×
∥
G
+
2
W
(
Λ
+
Λ
⊤
)
∥
n
u
c
l
e
a
r
.
(†)
​
  
= 
∥A∥ 
spectral
​
 ≤η
min
​
  
Λ
max
​
 traceG 
⊤
 A+traceΛ 
⊤
 (A 
⊤
 W+W 
⊤
 A)
= 
∥A∥ 
spectral
​
 ≤η
min
​
  
Λ
max
​
 traceA 
⊤
 (G+2W(Λ+Λ 
⊤
 ))
= 
Λ
max
​
  
∥A∥ 
spectral
​
 ≤η
min
​
 traceA 
⊤
 (G+2W(Λ+Λ 
⊤
 ))
= 
Λ
max
​
 −η×∥G+2W(Λ+Λ 
⊤
 )∥ 
nuclear
​
 .
​
  
​
 
Equation (1) reformulates the problem as a saddle point problem: the maximization over 
Λ
Λ will send the objective to infinity whenever the tangent space condition is violated. Equation (2) follows by applying properties of the trace and equation (3) follows from Sion’s minimax theorem. The inner minimization in equation (3) is solved by setting 
A
o
p
t
(
Λ
)
=
−
η
×
msign
⁡
(
G
+
2
W
(
Λ
+
Λ
⊤
)
)
A 
opt
​
 (Λ)=−η×msign(G+2W(Λ+Λ 
⊤
 )) where 
msign
⁡
msign is the matrix sign function.The matrix sign function snaps the singular values of a matrix to one. It may be computed efficiently on GPUs via Newton-Schulz iteration or the recent Polar Express algorithm. And we obtain equation (4) by substituting this expression for 
A
o
p
t
(
Λ
)
A 
opt
​
 (Λ) into equation (3). Equation (4) is known as the “dual problem” to 
(
†
)
(†) and we can solve it by gradient ascent. After some work, the gradient of the dual function is given by:

H
(
Λ
)
:
=
−
η
×
∇
Λ
∥
G
+
W
(
Λ
+
Λ
⊤
)
∥
n
u
c
l
e
a
r
=
−
η
×
[
W
⊤
m
s
i
g
n
(
G
+
2
W
(
Λ
+
Λ
⊤
)
)
+
msign
⁡
(
G
+
2
W
(
Λ
+
Λ
⊤
)
)
⊤
W
]
,
H(Λ)
​
  
:=−η×∇ 
Λ
​
 ∥G+W(Λ+Λ 
⊤
 )∥ 
nuclear
​
 
=−η×[W 
⊤
 msign(G+2W(Λ+Λ 
⊤
 ))+msign(G+2W(Λ+Λ 
⊤
 )) 
⊤
 W],
​
  
​
 
where the nuclear norm 
∥
⋅
∥
n
u
c
l
e
a
r
∥⋅∥ 
nuclear
​
  measures the sum of the singular values of a matrix.

Finally, we can write down the manifold Muon algorithm:Note that this algorithm is closely related to Jianlin Su’s solution. Where we run dual ascent, Jianlin’s solution amounts to solving for the maximum of the dual function 
H
(
Λ
)
=
0
H(Λ)=0 via a fixed point iteration.

Run gradient ascent on the dual variable 
Λ
←
Λ
+
α
×
H
(
Λ
)
Λ←Λ+α×H(Λ) to solve for 
Λ
o
p
t
Λ 
opt
​
 .
Compute the update 
A
o
p
t
=
−
η
×
msign
⁡
(
G
+
2
W
(
Λ
o
p
t
+
Λ
o
p
t
⊤
)
)
A 
opt
​
 =−η×msign(G+2W(Λ 
opt
​
 +Λ 
opt
⊤
​
 )).
Apply the update to the weights 
W
←
W
+
A
o
p
t
W←W+A 
opt
​
 .
Retract the weights back to the manifold 
W
←
msign
⁡
(
W
)
W←msign(W).
We ran a very small experiment to sanity check the algorithm and provide a minimal implementation for students or researchers to play with. Each training run finishes in less than a minute. The code is here and see Figure 6 for the setup and results.

Training a small MLP for 3 epochs on the CIFAR-10 dataset. The different lightly shaded blue curves show different weight decay settings for AdamW. Results were averaged over 3 random seeds. The manifold Muon optimizer attained higher train and test accuracy than AdamW. The third plot shows the final singular value distribution of the first weight matrix for the best performing learning rate: the singular values after training with manifold Muon are all close to 1. Manifold Muon increased the wall clock time per step compared to AdamW, although this could be improved by running fewer steps of dual ascent or adding momentum to the algorithm and running dual ascent online. Depending on other systems bottlenecks, the overhead may not be an issue.
Modular manifolds#
So far in this post, we have discussed manifold constraints for individual parameter tensors and co-designed optimization logic for these constraints. A question we have not answered is: what happens when we combine layers to build networks? Can we think about individual layers in isolation—or do we need to be careful about interactions between layers and modify the optimization logic in response? The goal of this section is to point out that there is a way to extend the reasoning we introduced in the previous two sections to the case of whole networks, and we call this the theory of modular manifolds.The theory of modular manifolds builds on research I did with my friend Tim Large, my postdoc advisor Phillip Isola, my PhD advisor Yisong Yue and many other amazing collaborators. At the end of the section, we provide some links to learn more.

The idea of modular manifolds is to build an abstraction that tells us how to budget learning rates across layers. The actual optimization logic in each layer ends up being the same as what we already worked out, except that the learning rate for a layer is modified depending on where the layer appears in the network. The abstraction rests upon a key observation made in our paper on the modular norm, that budgeting learning rates—both across layers and when scaling up individual layers—is intimately tied to understanding the Lipschitz sensitivity of the network output with respect to the weights. The abstraction tracks this sensitivity as we build the network, and manifold constraints help us get a much tighter understanding of this sensitivity.

The starting point for the abstraction is to think of any neural network module—from a layer to a whole transformer—as a mathematical object with three attributes:

A forward function 
f
:
W
×
X
→
Y
f:W×X→Y that maps from a parameter space 
W
=
R
d
W=R 
d
  and an input space 
X
X to an output space 
Y
Y.
A submanifold of the weight space 
M
⊂
W
M⊂W that the weights are constrained to.
A norm 
∥
⋅
∥
:
W
→
R
∥⋅∥:W→R that acts as a measuring stick on weight space.
For example, a linear module equipped with the spectral norm and constrained to the Stiefel manifold, for which we have already worked out an optimizer, would be written:

S
t
i
e
f
e
l
L
i
n
e
a
r
=
{
(
W
,
x
)
↦
W
x
,
(forward function)
S
t
i
e
f
e
l
(
m
,
n
)
,
(manifold)
∥
⋅
∥
s
p
e
c
t
r
a
l
.
(norm)
StiefelLinear= 
⎩
⎨
⎧
​
  
(W,x)↦Wx,
Stiefel(m,n),
∥⋅∥ 
spectral
​
 .
​
  
(forward function)
(manifold)
(norm)
​
 
Provided that an input 
x
x to the 
S
t
i
e
f
e
l
L
i
n
e
a
r
StiefelLinear module has unit 
ℓ
2
ℓ 
2
​
  norm, then 
S
t
i
e
f
e
l
L
i
n
e
a
r
StiefelLinear is Lipschitz with respect to its weights in the module’s assigned norm with Lipschitz constant one:This argument can be extended to the RMS norm on the input and the RMS–RMS operator norm on the weights.

∥
(
W
+
Δ
W
)
x
−
W
x
∥
2
≤
∥
Δ
W
∥
s
p
e
c
t
r
a
l
×
∥
x
∥
2
=
∥
Δ
W
∥
s
p
e
c
t
r
a
l
.
∥(W+ΔW)x−Wx∥ 
2
​
 ≤∥ΔW∥ 
spectral
​
 ×∥x∥ 
2
​
 =∥ΔW∥ 
spectral
​
 .
This type of Lipschitz statement helps us understand how to scale weight updates to this module since it gives us a bound on how much the output can change when we perturb the weights. But when we compose two modules, can we automatically compile a Lipschitz statement on the joint weight space of the new module? The answer turns out to be yes, if we follow special rules for building the new module:

The new forward function 
f
3
f 
3
​
  is given by composing the two existing forward functions 
f
1
f 
1
​
  and 
f
2
f 
2
​
 :
f
3
(
(
w
1
,
w
2
)
,
x
)
:
=
f
2
(
w
2
,
f
1
(
w
1
,
x
)
)
.
f 
3
​
 ((w 
1
​
 ,w 
2
​
 ),x):=f 
2
​
 (w 
2
​
 ,f 
1
​
 (w 
1
​
 ,x)).
The new manifold constraint 
M
3
M 
3
​
  is just the Cartesian product (see Figure 7 for a fun example) of the two existing manifolds 
M
1
M 
1
​
  and 
M
2
M 
2
​
 :
M
3
=
M
1
×
M
2
.
M 
3
​
 =M 
1
​
 ×M 
2
​
 .
The new norm function is the max of the two existing norm functions weighted by special scalar coefficients 
s
1
s 
1
​
  and 
s
2
s 
2
​
 . Letting 
∥
⋅
∥
1
∥⋅∥ 
1
​
  denote the first module’s norm and 
∥
⋅
∥
2
∥⋅∥ 
2
​
  denote the second module’s norm, the new norm 
∥
⋅
∥
3
∥⋅∥ 
3
​
  is given by:
∥
(
w
1
,
w
2
)
∥
3
:
=
max
⁡
(
s
1
⋅
∥
w
1
∥
1
,
s
2
⋅
∥
w
2
∥
2
)
.
∥(w 
1
​
 ,w 
2
​
 )∥ 
3
​
 :=max(s 
1
​
 ⋅∥w 
1
​
 ∥ 
1
​
 ,s 
2
​
 ⋅∥w 
2
​
 ∥ 
2
​
 ).
When we use this composite norm to derive optimizers—following the same recipe we used in the first two sections of this post—we end up deriving separate optimizers for each layer, but the scalar coefficients 
s
i
s 
i
​
  budget the learning rates across layers.

We give much more detail on this construction, including extending it to other ways of combining modules, in our paper on the modular norm—although the paper does not cover manifold optimization. You can also check out our paper on modular duality for more on building optimizers in the modular norm. The Modula project builds toward a programmatic implementation of this construction.

=
x
The Cartesian product is a simple way to glue together two manifolds. For example, the product of a line and a disk is a cylinder. We get one copy of the disk at every point on the line.
Directions for future work#
We are excited about any research that tries to make neural network training as principled and automatic as the forward pass. The ideas in this post benefitted strongly from interactions with external researchers like Jianlin Su and Franz Louis Cesista. We would love to see more work on these topics from the community.

Some possible directions for future work are:

Modularity. What manifolds should attention heads live on? Should embeddings be constrained differently than unembeddings? We can mix-and-match constraints in different parts of the network, or leave some tensors unconstrained.
Numerics. Manifold constraints also place constraints on the range of values that individual weight entries can take. Does this impact numerics, or make low-precision training easier?
Convex optimization. The manifold Muon algorithm involves running dual ascent. Can we apply more sophisticated convex optimization techniques to solve the dual problem faster or more reliably?
Convergence analysis. How fast do these algorithms converge? Does good conditioning of the weight matrices benefit convergence? Is there more that we can say theoretically?
Regularization. Manifold constraints implicitly regularize the model. Could we design constraints or tune their radii to improve generalization?
Architecture-optimizer co-design. While hard manifold constraints may not ultimately be the right way to constrain weight matrices, they exemplify the idea of tightly co-designing optimization algorithms with architecural components. Are there more opportunities here?
Non-Riemannian geometry. Most work on manifold optimization works in a Riemannian world where distances are induced by inner products and norm balls are ellipsoids. But neural networks are different: matrices act as operators, and operator norms like the spectral norm do not emerge from inner products. This implies, for example, that norm balls can have sharp corners and there is no unique gradient flow. Is there more to be discovered in this non-Riemannian world?
Practical implementation. Applying these techniques at scale requires efficient manifold operations on GPUs. The recent Polar Express paper shows promise for fast matrix sign computation. What other algorithmic innovations do we need?
Further reading#
Manifold optimization. Absil, Mahony & Sepulchre’s textbook is a standard reference. For the Stiefel manifold specifically, see Edelman et al, 1998. These works live in a Riemannian world. Similarly most machine learning papers that consider optimization on the Stiefel manifold take a Riemannian point of view: see Li et al, 2020, Kong et al, 2022 and Park et al, 2025 for some examples.

Non-Riemannian geometry in machine learning. Thomas Flynn’s paper from 2017 on duality structure gradient descent characterizes the neural network weight space as a Finsler manifold, meaning a manifold equipped with a norm. It is well worth a read. Also see Jianlin Su’s recent blog post on Stiefel Muon as well as Franz Louis Cesista’s blog post on a heuristic solution to Muon on the Stiefel manifold. Franz also wrote a followup blog post generalizing the solution presented here. The Scion paper imposes weight constraints a different way via convex combinations and Carlson et al, 2015 wrote an early paper on (unconstrained) spectral descent.

The Modula project. The goal of the Modula project is to build a library that automatically compiles steepest descent optimizers along with Lipschitz statements for general architectures. Check out the project page at https://modula.systems as well as our paper on the modular norm and modular duality. Our optimization anthology also provides an accessible route into this space of ideas.

Lipschitz-constrained deep learning. There has been a lot of work on this topic. For example, check out Louis Béthune and Tsui-Wei Weng’s PhD theses. Usually work on this topic does not connect weight-Lipschitzness to optimizer design. See also Anil et al, 2018 and our paper Newhouse et al, 2025.

Citation#
Please cite this work as:

Jeremy Bernstein, "Modular Manifolds",
Thinking Machines Lab: Connectionism, Sep 2025.

Or use the BibTeX citation:

@article{bernstein2025manifolds,
  author = {Jeremy Bernstein},
  title = {Modular Manifolds},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/modular-manifolds/},
  doi = {10.64434/tml.20250926}
}

prev
back to top
next
Thinking Machines Lab © 2025
·
Terms of service
·
Privacy notice