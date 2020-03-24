# Clifftrace

## Overview:
This project uses [Conformal Geometric Algebra (CGA)][cga] to build a ray tracing engine with the [clifford][clifford] library with the following functionality:
* Recursive tracing of reflected rays up to a specified max depth
* A Blinn-Phong lighting model
* Shadows
* Various CGA primitives and surfaces
* Multiple point light source

This project was originally built by [Sushant Achawal](https://github.com/sushachawal) as part of his masters project and is now used for benchmarking clifford and experimenting with CGA surfaces.

Here are a couple of the latest rendering from the code in the repo:
<figure>
<img src="https://github.com/hugohadfield/raytrace/blob/master/Combined.png?raw=true" alt="drawing" width="100%"/>
</figure>
<figure>
<img src="https://github.com/hugohadfield/raytrace/blob/master/fig.png?raw=true" alt="drawing" width="100%"/>
</figure>

## Splines
The code can also generate splines of various types through different objects. Here are some N'th order Bezier curves:
Order 1          |  Order 2 |Order 3
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/hugohadfield/raytrace/blob/master/n1.png?raw=true)  |  ![](https://github.com/hugohadfield/raytrace/blob/master/n2.png?raw=true) |  ![](https://github.com/hugohadfield/raytrace/blob/master/n3.png?raw=true)

It can also generate meshed and textured .obj files with correct vertex normals. Here is a KB spline through control circles with a checkerboard texture map applied:
<figure>
<img src="https://github.com/hugohadfield/raytrace/blob/master/horn.png?raw=true" alt="horn shape" width="100%"/>
</figure>


## Usage:

Run the script with `python3 clifftrace.py`
### Package requirements:

The script requires the following non-standard python packages:
* [clifford][clifford]
* [NumPy][NumPy]
* [pyganja][pyganja]
* [Numba][numba]

## Read about Geometric Algebra!

[Geometric Algebra][ga] is a super exciting field for exploring 3D geometry and beyond. For further reading into GA see:


* Many of the concepts used in the ray tracer can be found in *A Covariant Approach to Geometry using Geometric Algebra* which can be found online [here][CovApp]. The report really summarises the power of working in the conformal model.

* For a more complete introduction to GA check out *Geometric Algebra for Physicists* and for a deeper look into GA theory: *Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry* [(companion site here)][GAforCompSci] which includes documentation of another ray tracer implemented in GA!


[cga]: https://en.wikipedia.org/wiki/Conformal_geometric_algebra
[clifford]: https://github.com/pygae/clifford
[NumPy]: https://github.com/numpy/numpy
[matplotlib]: https://github.com/matplotlib/matplotlib
[Numba]: https://github.com/numba/numba
[ga]: https://en.wikipedia.org/wiki/Geometric_algebra
[CovApp]: http://www2.montgomerycollege.edu/departments/planet/planet/Numerical_Relativity/GA-SIG/Conformal%20Geometry%20Papers/Cambridge/Covarient%20Approach%20to%20Geometry%20Using%20Geometric%20Algebra.pdf
[GAforCompSci]: http://www.geometricalgebra.net/
[pyganja]: https://github.com/pygae/pyganja
[GAOnline]: https://github.com/hugohadfield/GAonline
