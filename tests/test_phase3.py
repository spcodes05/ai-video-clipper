"""Phase 3: layout, framing modes, silence/jump cuts, zoom, captions, hooks, presets, QA, e2e."""
import re
import shutil
import subprocess

import numpy as np
import pytest

from phase2.boundaries import Word
from phase2.faces import FaceBox
from phase3.captions import STYLE_SPECS, build_ass, caption_box_inside, plan_placement
from phase3.config import (CAPTION_STYLES, EDITING_STYLES, Phase3Cfg, Phase3ConfigError,
                           FramingCfg, ForegroundCfg, GroupCfg, OutputCfg, SilenceCfg, ZoomCfg)
from phase3.filters import GraphSpec, build_filter_complex
from phase3.framing import decide_mode, face_stats
from phase3.hooks import generate_hook
from phase3.jumpcuts import plan_jump_cuts
from phase3.layout import compute_layout, contain_size
from phase3.silence import detect_silences, frame_db, plan_silence, validate_keep
from phase3.timeline import complement, map_interval, retime
from phase3.zoom import ZoomEvent, plan_zoom, zoom_at, zoom_expression

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg not installed")


def W(t, text, d=0.3):
    return Word(t, t + d, text)


def words_from(text, t0=0.3, step=0.35):
    out, t = [], t0
    for tok in text.split():
        out.append(W(t, tok))
        t += step
    return out


# ------------------------------------------------------------------ 1/2 layout
def test_square_panel_on_1080x1920():
    L = compute_layout(OutputCfg(resolution="1080x1920"), ForegroundCfg())
    assert (L.out_w, L.out_h, L.fg_w, L.fg_h, L.fg_x, L.fg_y) == (1080, 1920, 1080, 1080, 0, 420)
    assert L.is_square and L.out_h - L.fg_h == 840


def test_panel_position_and_offset_and_width():
    o = OutputCfg(resolution="1080x1920")
    assert compute_layout(o, ForegroundCfg(position="top")).fg_y == 0
    assert compute_layout(o, ForegroundCfg(position="bottom")).fg_y == 840
    assert compute_layout(o, ForegroundCfg(offset_y=-100)).fg_y == 320
    L = compute_layout(o, ForegroundCfg(width=900))
    assert (L.fg_w, L.fg_h, L.fg_x) == (900, 900, 90)


def test_canvas_is_exactly_9_16_and_ratio_override():
    assert OutputCfg(resolution="1080x1920").dims() == (1080, 1920)
    assert OutputCfg(aspect_ratio="9:16").dims() == (1080, 1920)
    assert OutputCfg(aspect_ratio="1:1", resolution="1080x1920").dims() == (1080, 1080)
    L = compute_layout(OutputCfg(), ForegroundCfg(aspect_ratio="4:5"))
    assert L.fg_w * 5 == L.fg_h * 4


# ------------------------------------------------------------------ 3 landscape in square
def test_landscape_fits_square_without_stretch():
    w, h = contain_size(1920, 1080, 1080, 1080)
    assert (w, h) == (1080, 608) and abs(w / h - 16 / 9) < 0.01
    assert contain_size(1080, 1920, 1080, 1080) == (608, 1080)


# ------------------------------------------------------------------ 4/5/6 framing
def fb(x, w=120, t=0.0):
    return FaceBox(t, x, 250, w, w, 3.0, 0.5)


def samples(boxes_per_sample):
    return [(i * 0.5, b) for i, b in enumerate(boxes_per_sample)]


def decide(mode, smp, sw=1920, sh=1080, **g):
    return decide_mode(ForegroundCfg(fit_mode=mode), GroupCfg(**g), smp, sw, sh, sh, 1.0, True)


def test_auto_group_falls_back_to_contain():
    smp = samples([[fb(100), fb(500), fb(900), fb(1300)]] * 10)
    d = decide("auto", smp)
    assert d.mode == "contain" and "group" in d.reason


def test_auto_single_speaker_crops_and_two_close_faces_crop():
    assert decide("auto", samples([[fb(900)]] * 10)).mode == "crop"
    d = decide("auto", samples([[fb(700), fb(900)]] * 10))
    assert d.mode == "crop" and d.group_crop


def test_auto_two_far_faces_and_no_faces_contain():
    assert decide("auto", samples([[fb(100), fb(1700)]] * 10)).mode == "contain"
    assert decide("auto", samples([[]] * 10)).mode == "contain"
    assert decide_mode(ForegroundCfg(), GroupCfg(), [], 1920, 1080, 1080, 1.0, False).mode == "contain"


def test_forced_modes():
    smp = samples([[fb(100), fb(500), fb(900)]] * 6)
    assert decide("crop", smp).mode == "crop"
    assert decide("contain", samples([[fb(900)]] * 6)).mode == "contain"
    assert face_stats(smp).p90_faces == 3


# ------------------------------------------------------------------ 7/8 silence
def test_silence_detection_synthetic():
    sr = 16000
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, sr * 2), rng.normal(0, 1e-4, sr * 2),
                        rng.normal(0, 0.1, sr)]).astype(np.float32)
    s = detect_silences(frame_db(x), -38.0, 0.6)
    assert len(s) == 1 and abs(s[0][0] - 2.0) < 0.05 and abs(s[0][1] - 4.0) < 0.05


def test_silence_removal_never_cuts_words_and_keeps_pause():
    ws = [W(0.1, "Hello", 0.5), W(0.7, "world.", 0.5), W(4.1, "Wow!", 0.4)]
    cfg = SilenceCfg(enabled=True, preserve_comedic_pauses=False)
    p = plan_silence(ws, 5.0, [(1.3, 4.0)], cfg)
    assert p.removed > 1.0
    assert validate_keep(ws, p.keep) == []
    kept_gap = sum(min(b, 4.1) - max(a, 1.2) for a, b in p.keep if b > 1.2 and a < 4.1)
    assert kept_gap >= cfg.keep_pause - 0.01          # natural pause survives


def test_comedic_pause_preserved_and_laughter_untouched():
    ws = [W(0.1, "So what did he say?", 0.9), W(3.0, "Nothing.", 0.5)]
    normal = plan_silence(ws, 4.0, [(1.1, 3.0)], SilenceCfg(preserve_comedic_pauses=False))
    comic = plan_silence(ws, 4.0, [(1.1, 3.0)], SilenceCfg(preserve_comedic_pauses=True))
    assert comic.removed < normal.removed and comic.preserved_comedic >= 1
    laugh = plan_silence(ws, 4.0, [(1.1, 3.0)], SilenceCfg(preserve_comedic_pauses=False),
                         laughter=[(1.5, 2.8)])
    assert laugh.removed == 0 or all(not (a < 2.8 and b > 1.5) for a, b in laugh.cuts)


def test_no_audio_data_uses_gaps_but_audio_with_no_silence_cuts_nothing():
    ws = [W(0.1, "Hi.", 0.4), W(4.0, "Bye.", 0.4)]
    assert plan_silence(ws, 5.0, [], SilenceCfg()).removed == 0       # audio says: not silent
    assert plan_silence(ws, 5.0, None, SilenceCfg()).removed > 1.0


def test_max_cut_ratio_cap():
    ws = [W(0.0, "a.", 0.2), W(5.0, "b.", 0.2), W(10.0, "c.", 0.2), W(15.0, "d.", 0.2)]
    p = plan_silence(ws, 15.5, [(0.3, 15.0)], SilenceCfg(max_cut_ratio=0.2, preserve_comedic_pauses=False))
    assert p.removed <= 0.2 * 15.5 + 1e-6


def test_jump_cuts_fillers_and_stutters_not_comedy_repeats():
    ws = words_from("so um you know what I I think we should go no no no never", step=0.5)
    p = plan_jump_cuts(ws, 8.0, __import__("phase3.config", fromlist=["JumpCutCfg"]).JumpCutCfg(enabled=True))
    removed = {ws[i].text for i in p.removed_idx}
    assert "um" in removed and "I" in removed and "no" not in removed
    keep = complement(p.cuts, 8.0)
    kept_words = [w for i, w in enumerate(ws) if i not in p.removed_idx]
    assert validate_keep(kept_words, keep) == []


def test_timeline_mapping():
    keep = [(0.0, 2.0), (4.0, 6.0)]
    assert retime(1.0, keep) == 1.0 and retime(5.0, keep) == 3.0
    assert map_interval(4.5, 5.5, keep) == (2.5, 3.5)


# ------------------------------------------------------------------ 9 zoom
def test_zoom_events_are_sparse_bounded_and_timed():
    ws = words_from("We were driving home for hours and nothing happened at all today. "
                    "Then suddenly the car stopped and I screamed!", step=0.4)
    cfg = ZoomCfg(min_score=0.9)
    ev = plan_zoom(ws, 12.0, cfg)
    assert ev, "an exclamation should earn a zoom"
    for e in ev:
        assert 1.0 < e.scale <= cfg.maximum_scale
        assert cfg.min_event_seconds - 1e-6 <= e.end - e.start <= cfg.max_event_seconds + 1e-6
        assert e.end <= 12.0 and e.start >= 0
    assert sum(e.end - e.start for e in ev) <= cfg.max_fraction * 12.0 + 1e-6
    assert plan_zoom(ws, 12.0, ZoomCfg(enabled=False)) == []
    assert plan_zoom(ws, 12.0, cfg, blocked=[(0, 12)]) == []
    e = ev[0]
    assert zoom_at(ev, e.start - 1, 0.35) == 1.0
    assert abs(zoom_at(ev, (e.start + e.end) / 2, 0.35) - e.scale) < 1e-9
    assert 1.0 < zoom_at(ev, e.start + 0.1, 0.35) < e.scale          # eased, not a jump


def test_zoom_expression_wellformed():
    ex = zoom_expression([ZoomEvent(1, 3, 1.08)], 0.35)
    assert ex.count("(") == ex.count(")") and "clip(" in ex
    assert zoom_expression([], 0.35) == "1.0000"


# ------------------------------------------------------------------ 10/11 captions
def _layout():
    return compute_layout(OutputCfg(resolution="1080x1920"), ForegroundCfg())


def _events(ass):
    return [l for l in ass.splitlines() if l.startswith("Dialogue") and ",Default," in l]


def _cs(t):
    h, m, s = t.split(":")
    return round((int(h) * 3600 + int(m) * 60 + float(s)) * 100)


def test_caption_word_sync_all_styles():
    ws = words_from("I thought he was joking about the whole thing", t0=1.0, step=0.4)
    cfg = Phase3Cfg.from_dict({}).captions
    L = _layout()
    for style in CAPTION_STYLES:
        cfg.style = style
        ass, st = build_ass(ws, cfg, L, plan_placement(L, cfg))
        ev = _events(ass)
        assert ev and st.events == len(ev)
        starts = [_cs(l.split(",")[1]) for l in ev]
        # every word start appears as an event start (per-word styles) or inside a group (static/karaoke)
        assert starts == sorted(starts) and starts[0] == 100
        assert all(_cs(l.split(",")[2]) > _cs(l.split(",")[1]) for l in ev)
        if STYLE_SPECS[style]["mode"] in ("word", "pill"):
            assert {round(w.start * 100) for w in ws} <= set(starts)


def test_karaoke_durations_sum_to_event_length():
    ws = words_from("one two three four", t0=0.5, step=0.4)
    cfg = Phase3Cfg.from_dict({"captions": {"style": "karaoke"}}).captions
    L = _layout()
    ass, _ = build_ass(ws, cfg, L, plan_placement(L, cfg))
    line = _events(ass)[0]
    total = sum(int(n) for n in re.findall(r"\\kf(\d+)", line))
    dur = _cs(line.split(",")[2]) - _cs(line.split(",")[1])
    assert abs(total - dur) <= 1


def test_caption_positioning_below_panel_inside_safe_area():
    L = _layout()
    cfg = Phase3Cfg.from_dict({}).captions
    p = plan_placement(L, cfg)
    assert p.position == "below" and p.y_top >= L.fg_bottom and p.y_bottom <= L.out_h - cfg.safe_bottom
    assert caption_box_inside(p, L, cfg) == []
    cfg.position = "over_panel"
    assert plan_placement(L, cfg).y_bottom <= L.fg_bottom
    cfg.position, cfg.y_ratio = "custom", 0.1
    assert plan_placement(L, cfg).y_top == int(0.1 * L.out_h)


def test_caption_placement_scales_with_resolution_and_falls_back_when_no_room():
    small = compute_layout(OutputCfg(resolution="540x960"), ForegroundCfg())
    p = plan_placement(small, Phase3Cfg.from_dict({}).captions)
    assert p.position == "below" and p.y_bottom <= small.out_h
    tall = compute_layout(OutputCfg(resolution="1080x1920"), ForegroundCfg(position="bottom"))
    assert plan_placement(tall, Phase3Cfg.from_dict({}).captions).fallback


def test_caption_font_shrinks_for_long_words():
    ws = [W(0.0, "internationalization"), W(0.4, "responsibilities")]
    cfg = Phase3Cfg.from_dict({}).captions
    L = _layout()
    ass, _ = build_ass(ws, cfg, L, plan_placement(L, cfg))
    assert "\\fs" in ass


# ------------------------------------------------------------------ 12 hook
def test_hook_uses_only_spoken_words_and_is_short():
    ws = words_from("Okay so listen. I thought he was joking when he said that to me. Then it got worse.")
    h = generate_hook(ws, [], Phase3Cfg.from_dict({}).hook)
    spoken = {re.sub(r"[^\w']", "", w.text.lower()) for w in ws}
    toks = [re.sub(r"[^\w']", "", t.lower()) for t in h.text.replace("…", "").split()]
    assert set(toks) <= spoken and len(toks) <= 8 and h.source == "quote"


def test_hook_laughter_template_only_when_laugh_detected_and_manual_override():
    cfg = Phase3Cfg.from_dict({}).hook
    ws = words_from("ok then")
    assert generate_hook(ws, [], cfg).text != "WAIT FOR THE REACTION"
    assert generate_hook(ws, [(1, 3)], cfg, [0.9]).text == "WAIT FOR THE REACTION"
    cfg.text = "my own hook"
    assert generate_hook(ws, [], cfg).text == "MY OWN HOOK"
    cfg.enabled = False
    assert generate_hook(ws, [], cfg) is None


def test_hook_event_rendered_in_ass():
    L = _layout()
    cfg = Phase3Cfg.from_dict({})
    from phase3.hooks import Hook
    ass, st = build_ass(words_from("a b c"), cfg.captions, L, plan_placement(L, cfg.captions),
                        Hook("THIS GOT OUT OF HAND", "quote"), cfg.hook, hook_end=10)
    assert st.hook_event and "Style: Hook" in ass and "THIS GOT OUT" in ass


# ------------------------------------------------------------------ 13 presets / config
def test_presets_and_precedence():
    assert set(EDITING_STYLES) == {"podcast", "comedy", "interview", "general"}
    pod = Phase3Cfg.from_dict({"preset": "podcast"})
    com = Phase3Cfg.from_dict({"preset": "comedy"})
    itv = Phase3Cfg.from_dict({"preset": "interview"})
    assert pod.jump_cuts.enabled and not com.jump_cuts.enabled
    assert com.silence_removal.preserve_comedic_pauses and com.captions.style == "highlighted"
    assert itv.zoom.emphasis_scale < pod.zoom.emphasis_scale < com.zoom.emphasis_scale
    over = Phase3Cfg.from_dict({"preset": "podcast", "zoom": {"emphasis_scale": 1.1}})
    assert over.zoom.emphasis_scale == 1.1 and over.preset == "podcast"


def test_config_validation_errors():
    for bad in ({"preset": "nope"}, {"zoom": {"emphasis_scale": 2.0}}, {"bogus": 1},
                {"captions": {"style": "neon"}}, {"foreground": {"fit_mode": "stretch"}},
                {"captions": {"position": "custom"}}, {"output": {"aspect_ratio": "wide"}}):
        with pytest.raises(Phase3ConfigError):
            Phase3Cfg.from_dict(bad)
    assert Phase3Cfg.from_dict({"captions": {"style": "clean"}}).captions.style == "classic"


def test_cli_flags_enable_phase3_and_override(tmp_path):
    from main import parse_args
    from src.pipeline.config import load_config
    from src.pipeline.phase3_bridge import build_phase3_cfg
    a = parse_args(["--input", "x.mp4", "--editing-style", "comedy", "--fit-mode", "contain",
                    "--caption-style", "karaoke", "--hook-text", "WAIT", "--remove-silence"])
    assert a.editing_style == "comedy" and a.hook_text == "WAIT"
    cfg = load_config("config.yaml", {"phase3": True, "editing_style": a.editing_style, "fit_mode": a.fit_mode,
                                      "caption_style": a.caption_style, "hook_text": a.hook_text,
                                      "remove_silence": True})
    p = build_phase3_cfg(cfg)
    assert (p.preset, p.foreground.fit_mode, p.captions.style, p.hook.text) == ("comedy", "contain", "karaoke", "WAIT")
    assert build_phase3_cfg(load_config("config.yaml", {})) is None            # default: untouched


def test_filter_graph_is_balanced_and_square():
    L = _layout()
    g = GraphSpec(layout=L, src_w=1920, src_h=1080, fps=30, mode="crop", crop_w=1080, crop_h=1080,
                  crop_x_expr="420", keep=[(0, 2), (3, 5)], zoom_events=[ZoomEvent(0.5, 2.5, 1.08)],
                  reaction_windows=[(1, 2)], ass_name="captions.ass")
    fc = build_filter_complex(g)
    assert fc.count("[") == fc.count("]") and f"scale={L.fg_w}:{L.fg_h}" in fc
    assert "concat=n=2" in fc and "enable=" in fc and "[vout]" in fc and "[aout]" in fc


# ------------------------------------------------------------------ 14 end to end + QA
class FakeDetector:
    def __init__(self, boxes):
        import cv2
        self.cv2, self.boxes = cv2, boxes

    def detect(self, frame, t, analysis_width=640, next_frame=None):
        return [FaceBox(t, *b, 3.0, 0.5) for b in self.boxes]


@pytest.fixture(scope="module")
def src_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("p3") / "src.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30:duration=12", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=12,volume=enable='between(t,3.5,7.5)':volume=0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def _cfg(**over):
    d = {"preset": "podcast", "output": {"resolution": "540x960", "encoder": "cpu", "preset": "ultrafast"}}
    for k, v in over.items():
        d[k] = v
    return Phase3Cfg.from_dict(d)


def _words():
    return words_from("So I was telling him about the situation yesterday.", 0.3, 0.32) + \
        words_from("I thought he was joking!", 8.0, 0.35)


def _probe(p):
    import json
    o = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(p)],
                       capture_output=True, check=True).stdout
    return json.loads(o)["streams"]


@needs_ffmpeg
@pytest.mark.parametrize("mode,boxes,expect", [
    ("contain", [(900, 250, 150, 150)], "contain"),
    ("crop", [(900, 250, 150, 150)], "crop"),
    ("auto", [(100, 250, 120, 120), (500, 250, 120, 120), (900, 250, 120, 120)], "contain"),
    ("auto", [(900, 250, 150, 150)], "crop"),
])
def test_end_to_end_modes(tmp_path, src_video, mode, boxes, expect):
    from phase3.render import render_clip
    cfg = _cfg(foreground={"fit_mode": mode})
    out = tmp_path / "c.mp4"
    r = render_clip(src_video, out, 0, 11, _words(), cfg, detector=FakeDetector(boxes),
                    laughter=[(9.5, 10.8, 0.9)], report_dir=tmp_path / "rep")
    assert r.mode == expect and r.qa.ok
    st = _probe(out)
    v = next(s for s in st if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (540, 960) and any(s["codec_type"] == "audio" for s in st)
    assert (tmp_path / "rep" / "c.report.json").exists() and (tmp_path / "rep" / "c.contact.jpg").exists()
    assert r.report["cuts"]["removed_seconds"] > 1.0          # silence removed
    from phase3.silence import validate_keep as vk
    keep = r.report["cuts"]["segments"]
    assert vk([w for w in _words() if w.text.lower() != "um"], keep) == []


@needs_ffmpeg
def test_foreground_is_sharp_square_and_background_blurred(tmp_path, src_video):
    import cv2
    from phase3.render import render_clip
    out = tmp_path / "s.mp4"
    cfg = _cfg(foreground={"fit_mode": "crop"}, captions={"enabled": False}, hook={"enabled": False})
    render_clip(src_video, out, 0, 4, [], cfg, detector=FakeDetector([(600, 250, 150, 150)]))
    L = compute_layout(cfg.output, cfg.foreground)
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1500)
    ok, f = cap.read()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    lap = lambda a: cv2.Laplacian(a, cv2.CV_64F).var()
    assert ok and L.fg_w == L.fg_h == 540
    assert lap(g[L.fg_y:L.fg_bottom]) > 3 * max(lap(g[:L.fg_y]), 1.0)
    assert g[:L.fg_y].mean() > 10 and g[L.fg_bottom:].mean() > 10      # no black borders


@needs_ffmpeg
def test_qa_flags_wrong_size_video(tmp_path, src_video):
    from phase3.qa import check_clip
    cfg = _cfg()
    L = compute_layout(cfg.output, cfg.foreground)
    rep = check_clip(src_video, L, 12.0, cfg, {"src_has_audio": True, "words": []})
    assert not rep.ok and any("resolution" in i for i in rep.issues)
    assert not check_clip(tmp_path / "missing.mp4", L, 5, cfg, {}).ok


@needs_ffmpeg
def test_render_raises_on_failed_qa(tmp_path, src_video):
    from phase3.qa import Phase3QAError
    from phase3.render import render_clip
    cfg = _cfg(qa={"min_seconds": 50.0, "max_seconds": 60.0})
    with pytest.raises(Phase3QAError):
        render_clip(src_video, tmp_path / "x.mp4", 0, 6, _words(), cfg, detector=FakeDetector([(900, 250, 150, 150)]))


@needs_ffmpeg
def test_phase3_through_clip_renderer_and_pipeline_validation(tmp_path, src_video):
    from src.models.schemas import (AudioFeatures, CandidateClip, ClipAnalysis, RankedClip,
                                    Transcript, TranscriptSegment, WordTimestamp)
    from src.pipeline.config import AppConfig
    from src.pipeline.phase3_bridge import build_phase3_cfg
    from src.utils.ffmpeg import probe_video
    from src.video.qa import validate_clip
    from src.video.reframer import get_reframer
    from src.video.renderer import ClipRenderer
    cfg = AppConfig.model_validate({"video": {"width": 540, "height": 960, "preset": "ultrafast"},
                                    "phase3": {"enabled": True, "preset": "general",
                                               "output": {"encoder": "cpu"},
                                               "foreground": {"fit_mode": "contain"}}})
    p3 = build_phase3_cfg(cfg)
    ws = [WordTimestamp(word=w.text, start=w.start, end=w.end) for w in _words()]
    tr = Transcript(segments=[TranscriptSegment(start=0.3, end=3.5, text="x", words=ws[:9]),
                              TranscriptSegment(start=8.0, end=10, text="y", words=ws[9:])], duration=12)
    cand = CandidateClip(id="c", start=0, end=11, transcript="t", audio_features=AudioFeatures())
    ana = ClipAnalysis(candidate_id="c", humor=1, hook=1, surprise=1, emotion=1, shareability=1,
                       visual_interest=1, context_completeness=1, payoff=1, overall=1,
                       recommended_start=0, recommended_end=11, title="t", reasoning="r")
    clip = RankedClip(candidate=cand, analysis=ana, viral_score=50, final_start=0, final_end=11, clip_number=1)
    rend = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "analysis" / "captions",
                        phase3_cfg=p3)
    item = rend.render(src_video, probe_video(src_video), clip, tr, tmp_path / "clip_01.mp4")
    assert validate_clip(tmp_path / "clip_01.mp4", item.duration, 540, 960) == []
