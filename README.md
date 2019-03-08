# Clifftrace

## Overview:
This project uses [Conformal Geometric Algebra (CGA)][cga] to build a ray tracing engine with the [clifford][clifford] library with the following functionality:
* Recursive tracing of reflected rays up to a specified max depth
* A Blinn-Phong lighting model
* Shadows
* Spherical objects and planes **only**
* A single point light source

Here is the latest rendering from the code in the repo:

<figure>
<img src="https://github.com/sushachawal/CliffordRayTrace/blob/master/fig.png?raw=true" alt="drawing" width="100%"/>
</figure>

## Usage:

Run the script with `python3 clifftrace.py`
### Package requirements:

The script requires the following non-standard python packages:
* [clifford][clifford]
* [NumPy][NumPy]
* [Pillow][Pillow]

### Interface with GAOnline:

**Note GAOnline is now deprecated and a version is no longer hosted on a web server! Instructions still apply to a local version which can be forked from [here][GAOnline].**

The output image is saved in the working directory as `fig.png` but there is also a terminal output which allows the scene composition to be drawn in GAOnline so that it can be viewed interactively. An example terminal output is:
```
DrawLine((1.0^e245),rgb(255, 0 , 0));
DrawLine((0.70711^e245) - (0.70711^e345),rgb(255, 0 , 0));
DrawEucPoint(-(6.0^e2) + (6.0^e3) + (35.5^e4) + (36.5^e5),rgb(0, 0, 255));
...
```

Copy and paste the terminal output into the **box outlined in red** in the GAOnline example image shown below.

<figure>
<img src="https://github.com/sushachawal/CliffordRayTrace/blob/master/GAOnline.png?raw=true" alt="drawing" width="100%"/>
</figure>

The output draws:

1. **Camera:** The position, optic axis, viewing screen and corner rays.
2. **Objects in the scene:** Object geometry and colour.
3. **Lighting:** Position will be drawn as yellow point inside a sphere.

## Planned Future Work:

* Move interactive visualisation over to [pyganja][pyganja]
* ~~Acceleration with [Numba][Numba] starting with the `PointsFromPP` function.~~
* A front-end to interact with the view as in GAOnline. (With Tkinter? PyGame?)
* ~~Ability to draw planes.~~
* Ability to draw meshes.
* Full parallelisation either with [Numba][Numba] or on the GPU.
* Implement a [BSP-Tree][BSP] to accelerate intersection tests.
* Change the sphere intersection test to imitate that of the plane

## Read about Geometric Algebra!

[Geometric Algebra][ga] is a super exciting field for exploring 3D geometry and beyond. For further reading into GA see:

* [Mathoma's tutorials][YTtuts] are super cool and provide a good start. Skip to the 9th video if you have some basic understanding of linear algebra in 3D.

* Many of the concepts used in the ray tracer can be found in *A Covariant Approach to Geometry using Geometric Algebra* which can be found online [here][CovApp]. The report really summarises the power of working in the conformal model.

* For a more complete introduction to GA check out *Geometric Algebra for Physicists* and for a deeper look into GA theory: *Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry* [(companion site here)][GAforCompSci] which includes documentation of another ray tracer implemented in GA!

[GAOnline]: https://github.com/hugohadfield/GAonline
[cga]: https://en.wikipedia.org/wiki/Conformal_geometric_algebra
[clifford]: https://github.com/pygae/clifford
[NumPy]: https://github.com/numpy/numpy
[matplotlib]: https://github.com/matplotlib/matplotlib
[Numba]: https://github.com/numba/numba
[ga]: https://en.wikipedia.org/wiki/Geometric_algebra
[YTtuts]: https://www.youtube.com/watch?v=PNlgMPzj-7Q&list=PLpzmRsG7u_gqaTo_vEseQ7U8KFvtiJY4K
[CovApp]: http://www2.montgomerycollege.edu/departments/planet/planet/Numerical_Relativity/GA-SIG/Conformal%20Geometry%20Papers/Cambridge/Covarient%20Approach%20to%20Geometry%20Using%20Geometric%20Algebra.pdf
[GAforCompSci]: http://www.geometricalgebra.net/
[BSP]: https://en.wikipedia.org/wiki/Binary_space_partitioning
[Pillow]: https://github.com/python-pillow/Pillow
[pyganja]: https://github.com/pygae/pyganja
