r"""Bergman fans and complexes of order filters, as in AHK §2.

The combinatorial core is plain Python; SageMath is needed only for the
native-object conversion methods.  A ray is identified by a frozenset S,
representing e_S in Z^E / Z e_E.  In particular e_i and e_{\{i\}} are
one and the same ray, even when {i} belongs to the filter.
"""
from itertools import combinations
from operator import index as integer_index


def _subsets(items):
    items = tuple(items)
    for k in range(len(items) + 1):
        for choice in combinations(items, k):
            yield frozenset(choice)


class BergmanFilter:
    r"""The Chapter 2 Bergman fan and complex of an order filter.

    INPUT:

    - ``ground_set`` -- nonempty iterable of distinct hashable elements;
      its order fixes the coordinate system. Pass a list or tuple for
      reproducible ordering.
    - ``order_filter`` -- iterable of nonempty proper subsets of E. By
      default this must be the *entire* upward-closed filter.
    - ``generators`` -- if True, interpret ``order_filter`` as generators
      and take their upward closure inside the nonempty proper subsets.
    - ``basepoint`` -- element whose basis vector is eliminated in
      N_E = Z^E / Z e_E (default: first element of ground_set).

    The maximal cones are built by stellar subdivisions of the simplex
    fan (Proposition 2.4), rather than enumerating all compatible pairs.
    """

    def __init__(self, ground_set, order_filter=(), *, generators=False, basepoint=None):
        self.ground_set = tuple(ground_set)
        try:
            self.E = frozenset(self.ground_set)
        except TypeError as exc:
            raise ValueError('Ground-set elements must be hashable') from exc
        if not self.E or len(self.E) != len(self.ground_set):
            raise ValueError('The ground set must be nonempty and contain distinct elements')
        self.basepoint = self.ground_set[0] if basepoint is None else basepoint
        if self.basepoint not in self.E:
            raise ValueError('basepoint must be an element of the ground set')
        self.coordinate_elements = tuple(x for x in self.ground_set if x != self.basepoint)
        self._position = {x: i for i, x in enumerate(self.ground_set)}
        raw = set()
        for subset in order_filter:
            try:
                subset = frozenset(subset)
            except TypeError as exc:
                raise ValueError('Each filter member must be an iterable of ground-set elements') from exc
            if not subset or not subset < self.E:
                raise ValueError('Filter members must be nonempty proper subsets of E')
            raw.add(subset)
        if generators:
            self.order_filter = frozenset(
                subset for subset in _subsets(self.ground_set)
                if subset and subset != self.E and any(g <= subset for g in raw)
            )
        else:
            # It is enough to check one-element extensions.
            for subset in raw:
                for x in self.E - subset:
                    extended = subset | {x}
                    if extended != self.E and extended not in raw:
                        raise ValueError('order_filter is not upward closed: missing %r' % (extended,))
            self.order_filter = frozenset(raw)
        self._maximal_cache = None

    @property
    def dimension(self):
        """Fan dimension; the simplicial complex has dimension one less."""
        return len(self.ground_set) - 1

    def _key(self, subset):
        return (len(subset), tuple(self._position[x] for x in self.ground_set if x in subset))

    def ray_coordinates(self, subset):
        r"""Coordinates of e_S in the basis (e_i : i != basepoint)."""
        subset = frozenset(subset)
        if not subset <= self.E:
            raise ValueError('subset must be contained in the ground set')
        b = int(self.basepoint in subset)
        return tuple(int(x in subset) - b for x in self.coordinate_elements)

    def rays(self):
        """Ray labels in a deterministic order, with coincident rays identified."""
        if self.dimension == 0:
            return ()
        labels = {frozenset((x,)) for x in self.ground_set} | set(self.order_filter)
        return tuple(sorted(labels, key=self._key))

    def maximal_cones(self):
        """Return maximal cones as tuples of ray labels (frozensets)."""
        if self._maximal_cache is None:
            singletons = tuple(frozenset((x,)) for x in self.ground_set)
            # Facets of the boundary of the simplex, including the zero cone
            # in the one-element case.
            cones = {frozenset(s for s in singletons if s != omitted)
                     for omitted in singletons}
            # Add larger subsets first so that each intermediate family is a filter.
            for z in sorted(self.order_filter, key=lambda s: (-len(s), self._key(s)[1])):
                if len(z) == 1:
                    continue  # e_{\{i\}} = e_i: subdivision of a ray is trivial.
                face = frozenset(frozenset((x,)) for x in z)
                updated = set()
                for cone in cones:
                    if face <= cone:
                        for ray in face:
                            updated.add((cone - {ray}) | {z})
                    else:
                        updated.add(cone)
                cones = updated
            ordered = [tuple(sorted(cone, key=self._key)) for cone in cones]
            self._maximal_cache = tuple(sorted(ordered, key=lambda c: tuple(self._key(s) for s in c)))
        return self._maximal_cache

    def faces(self, dimension=None):
        """All faces as tuples of ray labels; the empty face has dimension -1.

        This enumerates faces and can be expensive. If dimension is supplied,
        only faces of that dimension are returned.
        """
        if dimension is not None:
            try:
                dimension = integer_index(dimension)
            except TypeError as exc:
                raise ValueError('dimension must be an integer >= -1') from exc
            if dimension < -1:
                raise ValueError('dimension must be an integer >= -1')
        result = set()
        for cone in self.maximal_cones():
            sizes = range(len(cone) + 1) if dimension is None else (dimension + 1,)
            for k in sizes:
                if k <= len(cone):
                    result.update(combinations(cone, k))
        return tuple(sorted(result, key=lambda f: (len(f), tuple(self._key(s) for s in f))))

    def f_vector(self):
        """Return (f_-1, f_0, ..., f_{n-1}) for the complex."""
        return tuple(len(self.faces(d)) for d in range(-1, self.dimension))

    def sage_fan(self):
        """Construct a Sage rational polyhedral fan in N_E (requires SageMath)."""
        from sage.all import Fan, ToricLattice
        rays = self.rays()
        index = {ray: i for i, ray in enumerate(rays)}
        cones = [tuple(index[ray] for ray in cone) for cone in self.maximal_cones()]
        return Fan(cones, rays=[self.ray_coordinates(ray) for ray in rays],
                   lattice=ToricLattice(self.dimension))

    def sage_complex(self):
        """Construct a Sage SimplicialComplex with integer vertices.

        Vertex i corresponds to self.rays()[i]. The empty face is retained
        even for a one-element ground set.
        """
        from sage.all import SimplicialComplex
        rays = self.rays()
        index = {ray: i for i, ray in enumerate(rays)}
        facets = [tuple(index[ray] for ray in cone) for cone in self.maximal_cones()]
        return SimplicialComplex(facets)

    def plot_data(self, kind="fan", *, radius=1):
        """Return plotting data for |E| <= 4, without requiring Sage.

        In a fan plot rays are truncated at Euclidean distance ``radius``;
        in a complex plot the vertices are the actual e_S coordinates.
        Coordinates are padded to two dimensions when dim(N_E) < 2.
        """
        from math import sqrt, isfinite
        if kind not in ("fan", "complex"):
            raise ValueError("kind must be 'fan' or 'complex'")
        if self.dimension > 3:
            raise ValueError("Visualization requires |E| <= 4")
        try:
            radius = float(radius)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("radius must be a finite positive number") from exc
        if not isfinite(radius) or radius <= 0:
            raise ValueError("radius must be a finite positive number")
        coords = {}
        for ray in self.rays():
            v = self.ray_coordinates(ray)
            if kind == "fan":
                length = sqrt(sum(x*x for x in v))
                v = tuple(radius*x/length for x in v)
            if len(v) < 2:
                v = tuple(v) + (0,) * (2-len(v))
            coords[ray] = v
        return {"kind": kind, "dimension": self.dimension,
                "vertices": coords, "facets": self.maximal_cones()}

    def plot(self, kind="fan", *, labels=True, radius=1, opacity=None):
        """Create a Sage graphic of the fan or complex for |E| <= 4.

        Fan cones are displayed as bounded truncations (not infinite cones).
        The complex uses the exact e_S vertices from Definition 2.2.
        ``labels`` controls ray labels; ``opacity`` is between 0 and 1.
        Return a 2D Sage Graphics object or, for |E|=4, a 3D object.
        """
        data = self.plot_data(kind, radius=radius)
        if opacity is None:
            opacity = 0.13 if data["dimension"] == 3 else 0.18
        try:
            opacity = float(opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("opacity must be between 0 and 1") from exc
        if not 0 <= opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")
        from sage.all import Graphics, line, point, polygon, text
        dim = data["dimension"]
        vertices = data["vertices"]
        origin = (0, 0, 0) if dim == 3 else (0, 0)
        palette = ("#4477AA", "#EE6677", "#228833", "#CCBB44",
                   "#66CCEE", "#AA3377", "#BBBBBB", "#EE7733")
        objects = []
        if dim == 3:
            from sage.all import line3d, point3d, polygon3d, text3d
        for i, facet in enumerate(data["facets"]):
            pts = [vertices[v] for v in facet]
            color = palette[i % len(palette)]
            if dim == 3:
                # The bounded cone is the tetrahedron conv(0, ray endpoints).
                triangles = ([pts] + [[origin, pts[j], pts[k]]
                             for j, k in combinations(range(len(pts)), 2)]
                             if kind == "fan" else [pts])
                for triangle in triangles:
                    if len(triangle) == 3:
                        objects.append(polygon3d(triangle, color=color, opacity=opacity))
            elif dim == 2 and kind == "fan" and len(pts) == 2:
                objects.append(polygon([origin] + pts, color=color, alpha=opacity))
        # Draw each edge exactly once. For fan plots include radial edges.
        for edge in self.faces(1):
            pts = [vertices[v] for v in edge]
            if dim == 3:
                objects.append(line3d(pts, color="#222222", thickness=2))
            elif dim >= 2:
                objects.append(line(pts, color="#222222", thickness=1.5))
        if kind == "fan":
            for v in vertices.values():
                if dim == 3:
                    objects.append(line3d([origin, v], color="#555555", thickness=1))
                else:
                    objects.append(line([origin, v], color="#555555", thickness=1.5))
            if dim < 3:
                objects.append(point(origin, color="black", size=22))
        for ray, v in vertices.items():
            label = "{" + ", ".join(str(x) for x in self.ground_set if x in ray) + "}"
            if dim == 3:
                objects.append(point3d(v, color="#111111", size=14))
                if labels:
                    objects.append(text3d(label, tuple(1.10*x for x in v), color="black"))
            else:
                objects.append(point(v, color="#111111", size=35))
                if labels:
                    offset = (0, 0.09) if dim == 1 else (0.05, 0.07)
                    objects.append(text(label, tuple(v[j] + offset[j] for j in range(2)),
                                        color="black", fontsize=9))
        # A zero-dimensional complex consists only of the empty face; it has
        # no vertices to draw. Return an empty 2D graphic in that case.
        if not objects:
            return Graphics()
        graphic = objects[0]
        for obj in objects[1:]:
            graphic += obj
        return graphic

    def plot_fan(self, **kwargs):
        """Plot the bounded visualization of the fan; see :meth:`plot`."""
        return self.plot("fan", **kwargs)

    def plot_complex(self, **kwargs):
        """Plot the complex with its original ray-vector vertices."""
        return self.plot("complex", **kwargs)

    def __repr__(self):
        return ('BergmanFilter(|E|=%d, |P|=%d, fan_dimension=%d)' %
                (len(self.E), len(self.order_filter), self.dimension))
