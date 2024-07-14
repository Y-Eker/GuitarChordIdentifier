import math


def calculate_slope(line):
    x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
    return math.inf if x2 - x1 == 0 else abs(y2 - y1 / x2 - x1)


def dist_two_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def similar_strong_line(line, strong_lines, slope_diff_threshold, point_diff_threshold) -> bool:
    line_x1, line_y1, line_x2, line_y2 = line[0][0], line[0][1], line[0][2], line[0][3]
    slope = calculate_slope(line)
    for strong_line in strong_lines:
        strong_line_x1, strong_line_y1, strong_line_x2, strong_line_y2 = strong_line[0][0], strong_line[0][1], \
            strong_line[0][2], strong_line[0][3]
        strong_slope = calculate_slope(strong_line)
        slope_similar = abs(slope - strong_slope) < slope_diff_threshold
        startpoint_similar = dist_two_points(strong_line_x1, strong_line_y1, line_x1, line_y1) < point_diff_threshold
        endpoint_similar = dist_two_points(strong_line_x2, strong_line_y2, line_x2, line_y2) < point_diff_threshold
        if slope_similar and startpoint_similar and endpoint_similar:
            return True
    return False
