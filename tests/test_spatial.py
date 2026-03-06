"""
tests/test_spatial.py
=====================
Tests for spatial weights construction and Moran's I diagnostics
as used in pipeline 08.

Tests use small synthetic geometries to avoid dependency on the
actual project data files, keeping the test suite fast and
self-contained.
"""
import numpy as np
import pytest
import geopandas as gpd
import libpysal
import esda
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Fixtures — synthetic spatial data
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_gdf():
    """
    A 3×3 grid of unit squares as a GeoDataFrame.
    Each cell shares boundaries with its neighbours (Queen contiguity).
    Total: 9 polygons.
    """
    polygons = []
    ids = []
    for row in range(3):
        for col in range(3):
            polygons.append(box(col, row, col + 1, row + 1))
            ids.append(f"cell_{row}_{col}")

    return gpd.GeoDataFrame({"id": ids, "geometry": polygons}, crs="EPSG:27700")


@pytest.fixture
def grid_weights(grid_gdf):
    """Row-standardised Queen contiguity weights for the 3×3 grid."""
    w = libpysal.weights.Queen.from_dataframe(
        grid_gdf, geom_col="geometry", use_index=True
    )
    w.transform = "r"
    return w


@pytest.fixture
def isolated_gdf():
    """
    Two polygons that do not share a boundary or corner.
    Both should be islands in a Queen weights matrix.
    """
    poly1 = box(0, 0, 1, 1)
    poly2 = box(5, 5, 6, 6)  # far from poly1, no shared boundary
    return gpd.GeoDataFrame(
        {"id": ["a", "b"], "geometry": [poly1, poly2]},
        crs="EPSG:27700"
    )


# ---------------------------------------------------------------------------
# Spatial weights tests
# ---------------------------------------------------------------------------

class TestQueenWeights:

    def test_weights_built(self, grid_weights):
        """Weights matrix should be built without error."""
        assert grid_weights is not None

    def test_correct_number_of_units(self, grid_weights):
        """Weights matrix should contain one entry per polygon."""
        assert grid_weights.n == 9

    def test_row_standardised(self, grid_weights):
        """
        Row-standardised weights must sum to 1.0 for every non-island unit.
        Uses the cardinalities to identify non-islands.
        """
        for i in range(grid_weights.n):
            neighbours = grid_weights.neighbors[i]
            if len(neighbours) > 0:
                row_sum = sum(grid_weights.weights[i])
                assert row_sum == pytest.approx(1.0, abs=1e-10)

    def test_corner_cell_has_three_neighbours(self, grid_weights, grid_gdf):
        """
        Corner cells in a 3×3 Queen grid have exactly 3 neighbours
        (the two adjacent cells and one diagonal).
        """
        # Cell (0,0) is index 0
        assert len(grid_weights.neighbors[0]) == 3

    def test_centre_cell_has_eight_neighbours(self, grid_weights):
        """
        The centre cell (1,1) in a 3×3 Queen grid has 8 neighbours —
        all surrounding cells including diagonals.
        """
        # Centre cell is index 4
        assert len(grid_weights.neighbors[4]) == 8

    def test_edge_cell_has_five_neighbours(self, grid_weights):
        """
        An edge (non-corner) cell has exactly 5 neighbours.
        Cell (0,1) is index 1.
        """
        assert len(grid_weights.neighbors[1]) == 5

    def test_symmetry(self, grid_weights):
        """
        If cell i is a neighbour of cell j, then j must be a neighbour
        of i (Queen contiguity is symmetric).
        """
        for i, neighbours in grid_weights.neighbors.items():
            for j in neighbours:
                assert i in grid_weights.neighbors[j]

    def test_islands_have_no_neighbours(self, isolated_gdf):
        """
        Polygons with no shared boundary or corner should have zero
        neighbours in the Queen weights matrix.
        """
        w = libpysal.weights.Queen.from_dataframe(
            isolated_gdf, geom_col="geometry", use_index=True
        )
        assert len(w.neighbors[0]) == 0
        assert len(w.neighbors[1]) == 0

    def test_untransformed_weights_not_row_standardised(self, grid_gdf):
        """
        Without row standardisation, weights are binary (0 or 1).
        Row sums equal the number of neighbours, not 1.
        """
        w = libpysal.weights.Queen.from_dataframe(
            grid_gdf, geom_col="geometry", use_index=True
        )
        # Centre cell (index 4) has 8 neighbours — row sum should be 8
        assert sum(w.weights[4]) == pytest.approx(8.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Moran's I tests
# ---------------------------------------------------------------------------

class TestMoransI:

    def test_perfectly_clustered_positive_autocorrelation(self, grid_gdf, grid_weights):
        """
        A perfectly spatially clustered variable (e.g. all high values
        in one corner, all low values in the other) should produce a
        strongly positive Moran's I close to 1.
        """
        # Values increase with column index — strong spatial pattern
        values = np.array([float(i % 3) for i in range(9)])
        mi = esda.Moran(values, grid_weights)
        assert mi.I > 0.3

    def test_checkerboard_negative_autocorrelation(self, grid_weights):
        """
        A checkerboard pattern (alternating high/low values) produces
        negative spatial autocorrelation — Moran's I should be negative.
        """
        values = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(9)])
        mi = esda.Moran(values, grid_weights)
        assert mi.I < 0.0

    def test_random_variable_insignificant(self, grid_weights):
        """
        A spatially random variable should not produce significant
        autocorrelation on average. Test over multiple seeds.
        """
        rng = np.random.default_rng(42)
        significant_count = 0
        n_trials = 50
        for _ in range(n_trials):
            values = rng.normal(size=9)
            mi = esda.Moran(values, grid_weights, permutations=199)
            if mi.p_sim < 0.05:
                significant_count += 1
        # Under the null, expect ~5% significant — allow up to 20% for
        # small sample variance
        assert significant_count / n_trials < 0.20

    def test_moran_i_range(self, grid_weights):
        """
        Moran's I is not strictly bounded to [-1, 1] but for typical
        spatial data should be well within that range.
        """
        rng = np.random.default_rng(0)
        values = rng.normal(size=9)
        mi = esda.Moran(values, grid_weights)
        assert -2.0 < mi.I < 2.0

    def test_moran_p_value_between_zero_and_one(self, grid_weights):
        """p-value from Monte Carlo simulation must be in [0, 1]."""
        values = np.arange(9, dtype=float)
        mi = esda.Moran(values, grid_weights, permutations=99)
        assert 0.0 <= mi.p_sim <= 1.0

    def test_constant_variable_moran_undefined(self, grid_weights):
        """
        A constant variable has zero variance. Moran's I is undefined
        in this case — libpysal returns NaN.
        """
        values = np.ones(9)
        mi = esda.Moran(values, grid_weights)
        assert np.isnan(mi.I)