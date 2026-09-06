"""Semantic-segmentation dataset from the clean frames (JVF-derived pseudo-GT) and a zero-shot benchmark.

Pipeline: areas (polygonize JVF boundaries) -> rasters (0.1 m faces/lines, 0.5 m DTM) -> point_labels
(uint8 per store point) -> render_labels (ERP 2000x1000 via point_id products) -> views (levelled gnomonic
tiles) -> dataset metadata; then models/fusion/bench/evaluate for the zero-shot comparison.
"""
