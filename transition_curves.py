"""두 긴 줄 사이 이동의 표시용 후보. 원본 CSV 또는 실행 궤적은 변경하지 않는다."""
import numpy as np


def lane_curve(previous, following, direction):
    """인접 도장 줄과 평행하게 따라가며 시작·끝을 연결한다 (스캔 좌표)."""
    previous, following = np.asarray(previous, float), np.asarray(following, float)
    a, b = previous[-1], following[0]
    ref = following if direction == 1 else previous
    normal = np.linalg.svd(ref - ref.mean(axis=0))[2][-1]
    def smooth(t):
        t = np.clip(t, 0., 1.)
        return t * t * (3 - 2 * t)
    if direction == 1:
        k = int(np.argmin(np.linalg.norm(following - a, axis=1)))
        if k < 2 or np.linalg.norm(following[k] - a) > .15:
            raise ValueError('다음 도장 줄과 연결할 수 없음')
        offset = a - following[k]
        lateral = (offset @ normal) * normal
        rest = offset - lateral
        idx = list(range(k - 1, -1, -1))
        nb = min(10, len(idx))
        points = [following[i] + lateral + (1 - smooth((n+1)/(nb+1))) * rest for n, i in enumerate(idx)]
    else:
        k = int(np.argmin(np.linalg.norm(previous - b, axis=1)))
        if k >= len(previous)-2 or np.linalg.norm(previous[k] - b) > .15:
            raise ValueError('이전 도장 줄과 연결할 수 없음')
        offset = b - previous[k]
        lateral = (offset @ normal) * normal
        rest = offset - lateral
        idx = list(range(len(previous)-1, k, -1))
        nb = min(10, len(idx))
        points = [previous[i] + lateral + smooth((n-(len(idx)-1-nb))/(nb+1)) * rest for n, i in enumerate(idx)]
    return np.vstack([a, points, b])
