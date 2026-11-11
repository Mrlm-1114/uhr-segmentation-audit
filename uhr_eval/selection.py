"""Validation selection, recovery and a linear-millisecond upper hull."""
from scripts.reproduce_tables import upper_hull

def select(ranked_candidates,latencies,budget):
    return next((r for r in ranked_candidates if r['path'] in latencies and latencies[r['path']]<=budget),None)

def recovery(measured_generic,measured_reference,published_best,published_reference):
    denominator=published_best-published_reference
    if denominator<=0:raise ValueError('Recovery denominator must be positive')
    return 100*(measured_generic-measured_reference)/denominator

def interpolate(hull,milliseconds):
    if not hull or not hull[0][0]<=milliseconds<=hull[-1][0]:return None
    for x,y,*_ in hull:
        if milliseconds==x:return y
    for a,b in zip(hull,hull[1:]):
        if a[0]<=milliseconds<=b[0]:return a[1]+(milliseconds-a[0])/(b[0]-a[0])*(b[1]-a[1])
