# Bergman fans of order filters (AHK §2)

Computes the fan Σ_P and simplicial complex Δ_P of Definition 2.3 of *Hodge Theory for Combinatorial Geometries*. This is the finite-set/order-filter construction of Chapter 2, **not** the matroidal Bergman fan of Chapter 3.

## Quick start

Run in SageMath from this directory (for a notebook, put this directory on Python's import path):

```python
from bergman import BergmanFilter

E = [0, 1, 2, 3]
# P is the upward closure of {0,1} inside the nonempty proper subsets of E.
B = BergmanFilter(E, [{0, 1}], generators=True)
print(B.order_filter)
print(B.rays())                # labels, each a frozenset
print(B.maximal_cones())       # maximal simplices/cones as ray labels
print(B.f_vector())            # (1, 7, 15, 10), starting with f_{-1}
print(B.faces(dimension=1))    # edges; dimension -1 gives the empty face

fan = B.sage_fan()             # Sage rational polyhedral fan
complex_ = B.sage_complex()    # Sage simplicial complex
print(fan.is_smooth())
print(complex_.dimension())
# Vertex i of the simplicial complex is the ray B.rays()[i].
```

If the full filter is already known, pass it directly; the default checks upward closure:

```python
B = BergmanFilter([0, 1, 2, 3], [{0, 1}, {0, 1, 2}, {0, 1, 3}])
```

The empty filter is `BergmanFilter(E)`. For the full filter, either list all nonempty proper subsets or use the singleton generators `BergmanFilter(E, [{x} for x in E], generators=True)` when `len(E) > 1`. For a one-element ground set, the full filter is empty.

## Visualization for small E

For `|E| <= 4`, plot either construction in SageMath:

```python
B = BergmanFilter([0, 1, 2, 3], [{0, 1}], generators=True)
fan_picture = B.plot_fan(labels=True, radius=1, opacity=0.12)
complex_picture = B.plot_complex(labels=True, opacity=0.4)
fan_picture.show()
complex_picture.show()
```

For three ground-set elements these are 2D Sage graphics. For four they are interactive 3D Sage graphics in a compatible Sage notebook/viewer (rotate the picture to inspect the facets). For two, the picture is placed on the horizontal axis; for one, the fan is a point and the complex has no geometric points. More than four elements raises `ValueError` rather than silently projecting.

The pictures have **different geometric meanings**:

* `plot_complex()` draws the actual simplices `conv(e_i, e_F)` from Definition 2.2, using the quotient-lattice coordinates. For a three-dimensional ambient space, the maximal simplices are translucent triangular faces.
* `plot_fan()` cannot draw infinite cones in full. Instead, each ray is truncated to length `radius` in the displayed coordinate system, and a maximal cone is illustrated by the convex hull of the origin and its truncated ray endpoints. These bounded pieces indicate the cones but are not the cones themselves. In 3D, translucent triangular surfaces outline those pieces.

The quotient-basis coordinates need not be orthonormal in any canonical metric, so Euclidean ray normalization is a plotting convention, not part of the paper's definition. The basepoint changes the coordinate picture. Labels are ray subsets, and can be hidden with `labels=False`. Transparency can be set between 0 and 1. Many overlapping transparent faces can obscure each other: reduce `opacity`, hide labels, or rotate the plot. Both methods return a Sage graphics object, so you can also use Sage's `show` or `save` methods as supported by your environment.

For custom renderers or inspection without Sage, `B.plot_data('fan', radius=1)` and `B.plot_data('complex')` return a dictionary containing the vertex-coordinate mapping and maximal facets. For fan data the endpoints are normalized; for complex data the original quotient-coordinate vertices are used.

## Conventions

* `ground_set` must be nonempty and have distinct, hashable elements. Use an ordered list/tuple for reproducible output. Ground-set elements need not be numbers or mutually comparable.
* A filter member must be a nonempty proper subset of E. `generators=True` takes upward closure; without it, a non-filter raises `ValueError`. The empty list is a valid filter or generator list.
* A ray label S denotes the vector e_S. Singleton labels serve both as element rays e_i and subset rays e_{ {i} }; they are **not** duplicated when a singleton is in P.
* Coordinates use the basis `e_i` for `i != basepoint` of `N_E = Z^E / Z e_E`. The default basepoint is the first ground-set element; it can be changed with `basepoint=...`. `B.coordinate_elements` records the basis order; `B.ray_coordinates(S)` gives e_S in this basis. Sage may internally reorder fan rays, so do not assume its ray indices match `B.rays()`; the complex's vertex labels do match.
* `B.maximal_cones()` returns rays of each maximal cone, and `B.faces()` returns their subsets. Since the cones are simplicial, the latter also describes all fan cones and all complex simplices. The empty tuple is the zero cone / empty face. For |E|=1, the fan is zero-dimensional and the complex consists of the empty face (dimension −1).
* The complex is represented abstractly by ray labels; its geometric realization uses the corresponding e_S vectors. The normal fan is represented in the quotient lattice, not in the original |E|-dimensional coordinate space.

## Method and limits

Start with the simplex fan (empty filter) and add filter members in decreasing cardinality, applying the stellar subdivision in Proposition 2.4. Adding a singleton is the identity. The implementation stores all maximal cones and may use exponential/factorial time and memory; for the full filter there are |E|! maximal cones. Constructing the upward closure and enumerating `faces()` or `f_vector()` can also be expensive. This is a first, transparent implementation, not yet a large-instance backend.

The combinatorial core uses only Python's standard library. SageMath is required for `sage_fan()` and `sage_complex()`; no extra packages are needed.

## Tests

```sh
sage -python -m unittest -v test_bergman
# or, to test the Sage-independent core only:
python -m unittest -v test_bergman
```

The tests compare all 166 order filters on a four-element E against an independent implementation of Definition 2.3, and check empty/full filter counts. Sage-native conversion tests run when Sage is importable. Sage was not available in the development environment, so the native conversion and rendering tests have not yet been executed there. Plot data and the plotting branches are tested without Sage, the latter using mock drawing primitives.
