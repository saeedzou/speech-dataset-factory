import math

def cut_by_speaker_label_V2(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.
    Uses a triangular-quantile target duration schedule and merges consecutive same-speaker segments
    until the target is reached (or the next merge would violate constraints).

    Args:
        vad_list (list): List of VAD segments with start, end, speaker labels (and any extra keys).

    Returns:
        list: A list of updated VAD segments after splitting/merging and filtering.
    """
    MERGE_GAP = 2.0
    MIN_SEGMENT_LENGTH = 3.0
    MAX_SEGMENT_LENGTH = 30.0

    TRI_LOW = 6.0
    TRI_MODE = 12.0
    TRI_HIGH = 27.0

    def tri_icdf(a, c, b, p):
        Fc = (c - a) / (b - a) if b > a else 0.5
        if p < Fc:
            return a + math.sqrt(max(0.0, p * (b - a) * (c - a)))
        return b - math.sqrt(max(0.0, (1 - p) * (b - a) * (b - c)))

    def make_targets(total_speech):
        mean = (TRI_LOW + TRI_MODE + TRI_HIGH) / 3.0
        n_est = max(1, int(round(total_speech / max(mean, 1e-9))))
        ps = [(i + 0.5) / n_est for i in range(n_est)]
        targets = [tri_icdf(TRI_LOW, TRI_MODE, TRI_HIGH, p) for p in ps]
        return [max(MIN_SEGMENT_LENGTH, min(t, MAX_SEGMENT_LENGTH)) for t in targets]

    if not vad_list:
        return []

    vad_list = sorted(vad_list, key=lambda x: (float(x["start"]), float(x["end"])))

    for i, v in enumerate(vad_list):
        if "index" not in v:
            v["index"] = i

    segs = []
    for vad in vad_list:
        s = float(vad["start"])
        e = float(vad["end"])
        if e < s:
            e = s
        dur = e - s

        if dur <= MAX_SEGMENT_LENGTH:
            segs.append(vad.copy())
            continue

        cur = s
        while (e - cur) > MAX_SEGMENT_LENGTH:
            part = vad.copy()
            part["start"] = cur
            part["end"] = cur + MAX_SEGMENT_LENGTH
            segs.append(part)
            cur += MAX_SEGMENT_LENGTH

        tail = vad.copy()
        tail["start"] = cur
        tail["end"] = e
        segs.append(tail)

    total_speech = sum(max(0.0, float(v["end"]) - float(v["start"])) for v in segs)
    targets = make_targets(total_speech)

    updated_list = []
    i = 0
    k = 0

    n = len(segs)
    while i < n:
        target = targets[min(k, len(targets) - 1)]

        base = segs[i].copy()
        spk = base.get("speaker")

        chunk_start = float(segs[i]["start"])
        chunk_end   = float(segs[i]["end"])
        j = i

        while (j + 1) < n:
            nxt = segs[j + 1]

            if nxt.get("speaker") != spk:
                break

            gap = float(nxt["start"]) - chunk_end
            if gap < 0:
                gap = 0.0

            if gap >= MERGE_GAP:
                break

            cand_end = max(chunk_end, float(nxt["end"]))
            cand_len = cand_end - chunk_start
            if cand_len > MAX_SEGMENT_LENGTH:
                break

            cur_len = chunk_end - chunk_start

            if cur_len < target:
                if cand_len <= target:
                    chunk_end = cand_end
                    j += 1
                    continue

                if abs(cand_len - target) <= abs(cur_len - target):
                    chunk_end = cand_end
                    j += 1
                break

            break

        base["start"] = chunk_start
        base["end"] = chunk_end
        updated_list.append(base)

        i = j + 1
        k += 1

    filter_list = [
        vad for vad in updated_list
        if float(vad["end"]) - float(vad["start"]) >= MIN_SEGMENT_LENGTH
    ]
    return filter_list

def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]

            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]


    return filter_list
