from src.data_prep.build_transient_masks import _tile_boxes


def test_tile_boxes_cover_image_with_overlap():
    w, h = 1320, 989
    boxes = _tile_boxes(w, h, (2, 3), overlap=64)
    assert len(boxes) == 6
    covered = [[False] * w for _ in range(2)]  # check row coverage at y=0 and y=h-1
    for (x0, y0, x1, y1) in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    # union covers every pixel column and row
    xs = sorted(boxes, key=lambda b: b[0])
    reach = 0
    for (x0, _, x1, _) in xs:
        assert x0 <= reach
        reach = max(reach, x1)
    assert reach == w
    ys = sorted(boxes, key=lambda b: b[1])
    reach = 0
    for (_, y0, _, y1) in ys:
        assert y0 <= reach
        reach = max(reach, y1)
    assert reach == h


def test_tile_boxes_1x1_is_full_image():
    assert _tile_boxes(100, 50, (1, 1), overlap=64) == [(0, 0, 100, 50)]
