"""Learned, lip-motion-based A/V sync detection.

Detects dialogue scenes (face detection plus speech activity overlap),
runs a pretrained SyncNet-family model per scene to get an offset
estimate and confidence, and fits a RANSAC regression across scenes to
classify title-level drift (constant offset, drift-early, drift-late, or
intermittent).
"""
