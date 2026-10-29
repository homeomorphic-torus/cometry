import unittest
from itertools import combinations
from math import factorial
from bergman import BergmanFilter, _subsets


def direct_faces(E, P):
    """Independent implementation of Definition 2.3, with ray deduplication."""
    E = frozenset(E)
    P = set(P)
    result = set()
    for flags in _subsets(P):
        flags = tuple(sorted(flags, key=len))
        if any(not a < b for a, b in zip(flags, flags[1:])):
            continue
        minimum = flags[0] if flags else E
        for I in _subsets(minimum):
            if I == minimum or I in P:
                continue
            result.add(frozenset([frozenset([i]) for i in I] + list(flags)))
    return result


class Tests(unittest.TestCase):
    def test_empty_and_full_filters(self):
        for n in range(1, 6):
            E = tuple(range(n))
            empty = BergmanFilter(E)
            self.assertEqual(len(empty.maximal_cones()), n)
            self.assertEqual(empty.f_vector(), tuple(__import__('math').comb(n, k) for k in range(n)))
            full = BergmanFilter(E, (s for s in _subsets(E) if s and len(s) < n))
            self.assertEqual(len(full.maximal_cones()), factorial(n))

    def test_all_filters_against_definition(self):
        E = tuple(range(4))
        proper = [s for s in _subsets(E) if s and len(s) < len(E)]
        count = 0
        for candidate in _subsets(proper):
            try:
                obj = BergmanFilter(E, candidate)
            except ValueError:
                continue
            actual = {frozenset(face) for face in obj.faces()}
            self.assertEqual(actual, direct_faces(E, candidate))
            count += 1
        self.assertEqual(count, 166)  # Dedekind number M(4) minus the two constant cases

    def test_invalid_and_generators(self):
        with self.assertRaises(ValueError):
            BergmanFilter([0, 1, 2, 3], [{0, 1}])
        b = BergmanFilter([0, 1, 2, 3], [{0, 1}], generators=True)
        self.assertEqual(b.order_filter, frozenset(map(frozenset, [{0, 1}, {0, 1, 2}, {0, 1, 3}])))
        self.assertEqual(b.ray_coordinates({0, 2}), (-1, 0, -1))
        self.assertEqual(len(b.maximal_cones()), 10)
        with self.assertRaises(ValueError):
            BergmanFilter([1, 1])
        with self.assertRaises(ValueError):
            BergmanFilter([0, 1], [{2}])

    def test_plot_data_and_validation(self):
        import math
        B = BergmanFilter([0, 1, 2, 3], [{0, 1}], generators=True)
        complex_data = B.plot_data("complex")
        self.assertEqual(complex_data["vertices"][frozenset({0, 1})], (0, -1, -1))
        fan_data = B.plot_data("fan", radius=2)
        for v in fan_data["vertices"].values():
            self.assertAlmostEqual(math.sqrt(sum(x*x for x in v)), 2)
        self.assertEqual(fan_data["facets"], B.maximal_cones())
        self.assertEqual(BergmanFilter([0]).plot_data("complex")["vertices"], {})
        with self.assertRaises(ValueError):
            B.plot_data("bad")
        with self.assertRaises(ValueError):
            B.plot_data(radius=0)
        with self.assertRaises(ValueError):
            BergmanFilter(range(5)).plot_data()

    def test_plot_primitive_generation_without_sage(self):
        # Mock only the rendering primitives; exercise our real geometry and branches.
        import sys, types
        from unittest.mock import patch
        class Graphic:
            def __init__(self, name="empty", *args, **kwargs):
                self.names = [name]
            def __add__(self, other):
                out = Graphic()
                out.names = self.names + other.names
                return out
        sage = types.ModuleType("sage")
        all_ = types.ModuleType("sage.all")
        for name in ("Graphics", "line", "point", "polygon", "text", "line3d",
                     "point3d", "polygon3d", "text3d"):
            setattr(all_, name, lambda *a, _name=name, **kw: Graphic(_name, *a, **kw))
        with patch.dict(sys.modules, {"sage": sage, "sage.all": all_}):
            for size in range(1, 5):
                obj = BergmanFilter(range(size))
                for kind in ("fan", "complex"):
                    g = obj.plot(kind)
                    self.assertTrue(g.names)
            three = BergmanFilter(range(4)).plot_complex(labels=False)
            self.assertIn("polygon3d", three.names)
            self.assertNotIn("text3d", three.names)
            with self.assertRaises(ValueError):
                BergmanFilter([0, 1]).plot(opacity=2)

    def test_sage_conversions_when_available(self):
        try:
            import sage.all  # noqa
        except ImportError:
            self.skipTest('SageMath unavailable')
        b = BergmanFilter([0, 1, 2, 3], [{0, 1}], generators=True)
        fan = b.sage_fan()
        complex_ = b.sage_complex()
        self.assertEqual(fan.dim(), 3)
        self.assertEqual(len(fan.generating_cones()), len(b.maximal_cones()))
        self.assertEqual(complex_.dimension(), 2)
        self.assertEqual(len(complex_.facets()), len(b.maximal_cones()))
        self.assertEqual(BergmanFilter([0]).sage_complex().dimension(), -1)
        for n in range(1, 5):
            obj = BergmanFilter(range(n))
            self.assertIsNotNone(obj.plot_fan())
            self.assertIsNotNone(obj.plot_complex())

if __name__ == '__main__':
    unittest.main()
