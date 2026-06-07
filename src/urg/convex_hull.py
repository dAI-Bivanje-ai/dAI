import numpy as np

epsilon = 1e-9


def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def polar_angle(p, origin):
    return np.arctan2(p[1] - origin[1], p[0] - origin[0])


def vector_angle(prev, current, candidate):
    v1 = (current[0] - prev[0], current[1] - prev[1])
    v2 = (candidate[0] - current[0], candidate[1] - current[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 < epsilon or mag2 < epsilon:
        return float('inf')
    cos_fi = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    return np.arccos(cos_fi)


def jarvis(points):
    pts = list(points)
    convex_hull = []

    min_p = pts[0]
    for point in pts:
        if point[1] < min_p[1]:
            min_p = point
        elif point[1] == min_p[1] and point[0] < min_p[0]:
            min_p = point

    s0 = min_p
    convex_hull.append(s0)
    pts.remove(s0)

    s1 = None
    best_angle = float('inf')
    best_dist = float('inf')
    for p in pts:
        a = polar_angle(p, s0)
        d = dist(p, s0)
        if a < best_angle or (abs(a - best_angle) < epsilon and d < best_dist):
            best_angle = a
            best_dist = d
            s1 = p
    convex_hull.append(s1)
    pts.remove(s1)

    while True:
        s_prev_2 = convex_hull[-2]
        s_prev = convex_hull[-1]

        best = None
        best_angle = float('inf')

        for point in pts + [s0]:
            ang = vector_angle(s_prev_2, s_prev, point)
            if ang < best_angle - epsilon:
                best_angle = ang
                best = point
            elif abs(ang - best_angle) < epsilon:
                if dist(point, s_prev) < dist(best, s_prev):
                    best = point

        if best is s0 or best is None:
            break

        convex_hull.append(best)
        pts.remove(best)

    return convex_hull


# graham

def find_noncollinear(points):
    p0, p1 = points[0], points[1]
    for p2 in points[2:]:
        cross = (p1[0]-p0[0]) * (p2[1]-p0[1]) - (p1[1]-p0[1]) * (p2[0]-p0[0])
        if abs(cross) > epsilon:
            return p0, p1, p2
    return None


def graham_angle(p, origin):
    angle = np.arctan2(p[1] - origin[1], p[0] - origin[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle


def graham_scan(points):
    result = find_noncollinear(points)
    if result is None:
        return []

    # težišče vseh točk
    ox = sum(p[0] for p in points) / len(points)
    oy = sum(p[1] for p in points) / len(points)
    O = (ox, oy)

    angle_pairs = []
    for p in points:
        angle_pairs.append((graham_angle(p, O), p))
    angle_pairs.sort()
    points_sorted = []
    for _, p in angle_pairs:
        points_sorted.append(p)

    s0 = points_sorted[0]
    best_dist = dist(points_sorted[0], O)
    for p in points_sorted[1:]:
        d = dist(p, O)
        if d > best_dist:
            best_dist = d
            s0 = p
    idx = points_sorted.index(s0)
    points_sorted = points_sorted[idx:] + points_sorted[:idx]

    hull = [points_sorted[0], points_sorted[1]]
    for pk in points_sorted[2:]:
        while len(hull) > 1:
            pi, pj = hull[-2], hull[-1]
            cross = (pj[0]-pi[0]) * (pk[1]-pi[1]) - (pj[1]-pi[1]) * (pk[0]-pi[0])
            if cross > epsilon:
                break
            hull.pop()
        hull.append(pk)

    while len(hull) > 2:
        pi, pj = hull[-2], hull[-1]
        pk = hull[0]
        cross = (pj[0]-pi[0]) * (pk[1]-pi[1]) - (pj[1]-pi[1]) * (pk[0]-pi[0])
        if cross > epsilon:
            break
        hull.pop()

    return hull


def triangle_area(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    ploscina = 0.5 * abs(x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)
    return ploscina


def triangle_biggest_surface(points, a, b):
    best = None
    best_area = -1
    for point in points:
        area = triangle_area(point, a, b)
        if area > best_area:
            best_area = area
            best = point
    return best


def point_side(a, b, p):
    return (b[0]-a[0]) * (p[1]-a[1]) - (b[1]-a[1]) * (p[0]-a[0])


def find_hull(points, a, b):
    if not points:
        return []

    best_surface = triangle_biggest_surface(points, a, b)

    left1 = []
    for p in points:
        if point_side(a, best_surface, p) > 0:
            left1.append(p)

    left2 = []
    for p in points:
        if point_side(best_surface, b, p) > 0:
            left2.append(p)

    return find_hull(left1, a, best_surface) + [best_surface] + find_hull(left2, best_surface, b)


def quickhull(points):
    p_min = points[0]
    p_max = points[0]
    for p in points[1:]:
        if p[0] < p_min[0]:
            p_min = p
        if p[0] > p_max[0]:
            p_max = p

    upper = []
    lower = []
    for point in points:
        side = point_side(p_min, p_max, point)
        if side > 0:
            upper.append(point)
        elif side < 0:
            lower.append(point)

    hull_upper = find_hull(upper, p_min, p_max)
    hull_lower = find_hull(lower, p_max, p_min)

    convex_hull = [p_min] + hull_upper + [p_max] + hull_lower
    return convex_hull
