# -*- coding: utf-8 -*-


def parse_region(region):
    """
    Args:
        region: 1-based region
    Returns:
        (chrom, start, end) 0-based coordinate
    """
    sp = region.split(':')
    chrom = sp[0]
    if len(sp) == 1:
        return (chrom, 0, None)
    sp = sp[1].split('-')
    start = int(sp[0])
    if len(sp) == 1:
        return (chrom, start, None)
    return (chrom, start - 1, int(sp[1]))
