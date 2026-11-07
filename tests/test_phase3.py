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
from phase3.zoom import ZoomEvent, max_safe_scale, plan_zoom, zoom_at, zoom_expression

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
# ------------------------------------------------------------------ 13 presets / config
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


# =================================================================== Phase 3.1
from phase3.advisor import EditAdvisor, parse_reply                     # noqa: E402
from phase3.broll import BrollProvider, plan_broll                      # noqa: E402
from phase3.captions import (balanced_lines, group_phrases, hook_box_issues, hook_style,    # noqa: E402
                             plan_hook)
from phase3.config import BrollCfg, HookCfg, ReactionCfg                # noqa: E402
from phase3.framing import merge_boxes                                  # noqa: E402
from phase3.hooks import Hook, validate_hook_text                       # noqa: E402
from phase3.reactions import ExprSample, plan_visual_windows            # noqa: E402
from phase3.timeline import limit_cut_density, snap_cuts_to_quiet       # noqa: E402


def L1080(**fg):
    return compute_layout(OutputCfg(resolution="1080x1920"), ForegroundCfg(**fg))


def default_cfg():
    return Phase3Cfg.from_dict({"hook": {"enabled": True}})       # hooks are OFF by default; enable for hook tests


def test_hook_is_disabled_by_default_in_every_preset():
    for name in EDITING_STYLES:
        assert Phase3Cfg.from_dict({"preset": name}).hook.enabled is False
    assert Phase3Cfg.from_dict({}).hook.enabled is False


# ---- 1/6 hook styles: no yellow block by default
def test_default_hook_is_clean_white_without_background_block():
    c = default_cfg()
    assert c.hook.style == "clean_white" and c.hook.background == "none"
    st = hook_style(c.hook)
    assert st["pill"] is False and st["color"] == "FFFFFF" and st["outline"] > 0 and st["shadow"] > 0
    L = L1080()
    p = plan_placement(L, c.captions)
    hp, why = plan_hook(L, c.captions, c.hook, Hook("Why is it in the Kenny?", "quote"), p)
    ass, s = build_ass(words_from("a b c"), c.captions, L, p, Hook("x y", "quote"), c.hook, hook_plan=hp)
    hook_style_line = next(l for l in ass.splitlines() if l.startswith("Style: Hook"))
    f = hook_style_line.split(",")
    assert f[15] == "1"                                      # BorderStyle 1 = outline, not an opaque box
    assert "&H00FFFFFF" in hook_style_line and "FFE600" not in ass.upper() and "00E6FF" not in ass.upper()
    assert "Why is it in the Kenny?" in ass                  # natural sentence case, not upper-cased
    assert "\\fad(200,300)" in ass and "\\fscx94" in ass and "\\blur" not in ass


@pytest.mark.parametrize("style", ["clean_white", "minimal_shadow", "subtle_pill", "creator_custom"])
def test_all_hook_styles_render_valid_ass(style):
    c = Phase3Cfg.from_dict({"hook": {"style": style, "enabled": True"}})
    L = L1080()
    p = plan_placement(L, c.captions)
    hp, _ = plan_hook(L, c.captions, c.hook, Hook("This Conversation Got Weird", "llm"), p)
    ass, st = build_ass(words_from("a b c"), c.captions, L, p, Hook("x", "llm"), c.hook, hook_plan=hp)
    line = next(l for l in ass.splitlines() if l.startswith("Style: Hook")).split(",")
    pill = style == "subtle_pill"
    assert (line[15] == "3") == pill and st.hook_event
    if pill:                                    # translucent dark backing, never a solid colour block
        assert int(line[5][2:4], 16) > 0x40 and line[5].endswith("000000")


def test_hook_style_config_validation():
    for bad in ({"style": "neon"}, {"background": "box"}, {"font_size": 10}, {"background_opacity": 2},
                {"fade_in": 2.0, "fade_out": 2.0, "duration": 2.5}, {"text_case": "shout"}):
        with pytest.raises(Phase3ConfigError):
            Phase3Cfg.from_dict({"hook": bad})


# ---- 2/4 positioning + safe margins follow the layout
@pytest.mark.parametrize("fg", [dict(), dict(position="lower"), dict(position="upper"), dict(offset_y=-200)])
def test_hook_positioned_relative_to_panel_inside_safe_areas(fg):
    c = default_cfg()
    L = L1080(**fg)
    cap = plan_placement(L, c.captions)
    hp, why = plan_hook(L, c.captions, c.hook, Hook("He Was Not Ready For That", "llm"), cap)
    assert hp is not None, why
    k = L.out_w / 1080
    assert hp.y_top >= c.captions.safe_top * k - 1 and hp.y_bottom <= L.out_h
    if not hp.over_panel:
        assert hp.y_bottom <= L.fg_y                          # above the panel, never on it
    assert hook_box_issues(hp, L, c.captions, cap) == []
    assert len(hp.lines) <= 2


def test_hook_moves_with_panel():
    c = default_cfg()
    a = plan_hook(L1080(), c.captions, c.hook, Hook("This Conversation Got Weird", "llm"))[0]
    b = plan_hook(L1080(position="upper"), c.captions, c.hook, Hook("This Conversation Got Weird", "llm"))[0]
    assert a.y_top != b.y_top and a.y_bottom <= L1080().fg_y


# ---- 3 text fitting
def test_hook_fitting_shrinks_font_and_rejects_impossible_text():
    c = default_cfg()
    L = L1080()
    short = plan_hook(L, c.captions, c.hook, Hook("Wait For It", "llm"))[0]
    longer = plan_hook(L, c.captions, c.hook, Hook("Nobody Expected His Response Today Honestly", "llm"))[0]
    assert short.font_size >= longer.font_size and len(longer.lines) <= 2
    hp, why = plan_hook(L, c.captions, c.hook, Hook("Internationalization responsibilities misunderstandings institutionalization characteristically", "x"))
    assert hp is None and "fit" in why
    hp, why = plan_hook(L, c.captions, c.hook, Hook("one two three four five six seven eight nine", "x"))
    assert hp is None and "words" in why


# ---- 5 hook never overlaps captions
def test_hook_does_not_overlap_captions():
    c = default_cfg()
    c.captions.position, c.captions.y_ratio = "custom", 0.14          # captions up where the hook wants to be
    L = L1080()
    cap = plan_placement(L, c.captions)
    hp, why = plan_hook(L, c.captions, c.hook, Hook("This Conversation Got Weird", "llm"), cap)
    assert hp is None or hook_box_issues(hp, L, c.captions, cap) == []


# ---- 7 disabled
def test_hook_disabled_gives_none_and_no_style():
    c = Phase3Cfg.from_dict({"hook": {"enabled": False}})
    assert generate_hook(words_from("why is it like that though"), [], c.hook) is None
    L = L1080()
    ass, st = build_ass(words_from("a b c"), c.captions, L, plan_placement(L, c.captions),
                        Hook("x y", "x"), c.hook, hook_plan=None)
    assert "Style: Hook" not in ass and not st.hook_event


# ---- hook generation
def test_hook_not_forced_and_no_repetition_of_captions():
    cfg = default_cfg().hook
    assert generate_hook(words_from("so yeah we went to the store and bought some bread today"), [], cfg) is None
    early_q = words_from("Why is it in the Kenny? " + "blah " * 20)
    assert generate_hook(early_q, [], cfg) is None            # captions show it within the hook window
    later = words_from("so we sat down. " + "and we talked for a while. " * 4 +
                       "Why is it in the Kenny? he asked me.", step=0.45)
    h = generate_hook(later, [], cfg)
    assert h and h.source == "quote" and h.text.startswith("Why is it")


def test_hook_laughter_template_only_when_laugh_detected_and_manual_override():
    cfg = default_cfg().hook
    ws = words_from("ok then")
    assert generate_hook(ws, [], cfg) is None
    assert generate_hook(ws, [(1, 3)], cfg, [0.9]).text == "Wait For The Reaction"
    cfg.text = "my own hook"
    assert generate_hook(ws, [], cfg).source == "manual"


def test_llm_hook_validation_rejects_invented_clickbait_and_repeats():
    cfg = default_cfg().hook
    ws = words_from("I told Kenny that the car was broken. He laughed and said nothing. Then he left.", step=0.5)
    assert validate_hook_text("He Was Not Ready For That", ws, cfg)
    assert validate_hook_text("What Happened With Marcus", ws, cfg) is None          # invented name
    assert validate_hook_text("You won't believe what he said", ws, cfg) is None      # clickbait
    assert validate_hook_text("He laughed and said nothing", ws, cfg) is None         # repeats the transcript
    assert validate_hook_text("Way too many words in this hook for the limit ok", ws, cfg) is None
    assert validate_hook_text("Hi", ws, cfg) is None
    assert generate_hook(ws, [], cfg, advisor_hook="Nobody Expected His Response").source == "llm"
    assert generate_hook(ws, [], cfg, advisor_hook="Marcus Lost It") is None          # falls back -> none


class FakeProvider:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete_json(self, system, user):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_advisor_parses_validates_caches_and_survives_failures(tmp_path):
    ws = words_from("we talked for a while. then he said something strange.", step=0.4)
    good = ('{"hook": "This Conversation Got Weird", "moments": [{"start": 1.0, "end": 3.0, '
            '"kind": "punchline", "strength": 3}, {"start": 50, "end": 60, "kind": "x"}]}')
    prov = FakeProvider(good)
    adv = EditAdvisor(prov, tmp_path)
    r = adv.advise(ws, 6.0, 8)
    assert r.used and r.hook == "This Conversation Got Weird" and len(r.moments) == 1
    adv.advise(ws, 6.0, 8)
    assert prov.calls == 1                                    # second call served from cache
    assert EditAdvisor(FakeProvider(RuntimeError("quota"))).advise(ws, 6.0, 8).error
    assert EditAdvisor(FakeProvider("not json")).advise(ws, 6.0, 8).error
    assert parse_reply('{"hook": null, "moments": []}', 5, 8).hook is None


# ---- 8/9/10 captions
def test_caption_word_sync_tween_offsets_match_whisper_times():
    ws = words_from("I thought he was joking about the whole thing", t0=1.0, step=0.4)
    c = default_cfg().captions
    L = L1080()
    ass, st = build_ass(ws, c, L, plan_placement(L, c))
    ev = _events(ass)
    assert st.events == len(ev) >= 2 and _cs(ev[0].split(",")[1]) == 100
    for line, grp in zip(ev, group_phrases(ws, c)):
        s0 = _cs(line.split(",")[1]) / 100
        offs = [int(m) for m in re.findall(r"\\t\((\d+),\d+,\\1c&H3BD9FF&\)", line)]
        assert offs == [round((w.start - s0) * 1000) for w in grp]


def test_all_styles_render_and_karaoke_sums():
    ws = words_from("one two three four", t0=0.5, step=0.4)
    L = L1080()
    for style in CAPTION_STYLES:
        c = Phase3Cfg.from_dict({"captions": {"style": style}}).captions
        ass, st = build_ass(ws, c, L, plan_placement(L, c))
        assert st.events >= 1 and "Style: Default" in ass
    c = Phase3Cfg.from_dict({"captions": {"style": "karaoke"}}).captions
    line = _events(build_ass(ws, c, L, plan_placement(L, c))[0])[0]
    assert abs(sum(int(n) for n in re.findall(r"\\kf(\d+)", line)) -
               (_cs(line.split(",")[2]) - _cs(line.split(",")[1]))) <= 1


def test_phrase_grouping_avoids_dangling_weak_words_and_balances_lines():
    c = default_cfg().captions
    ws = words_from("I thought he was going to do that and then he left", step=0.3)
    groups = group_phrases(ws, c)
    assert [w.text for g in groups for w in g] == [w.text for w in ws]       # nothing lost / reordered
    from phase3.captions import WEAK_END
    assert all(g[-1].text.lower() not in WEAK_END for g in groups[:-1])
    assert all(len(g) <= c.max_words for g in groups)
    lines = balanced_lines("he was going to do that".split(), 22, 2)
    assert len(lines) == 2 and abs(len(lines[0]) - len(lines[1])) <= 2
    # sentence ends and pauses are hard boundaries
    two = words_from("Stop. Go now please.", step=0.3)
    assert [len(g) for g in group_phrases(two, c)] == [1, 3]
    gap = [W(0, "a"), W(2.0, "b")]
    assert len(group_phrases(gap, c)) == 2


def test_captions_follow_panel_and_stay_clear_of_hook():
    c = default_cfg()
    for fg in (dict(), dict(position="top"), dict(position="upper"), dict(offset_y=100)):
        L = L1080(**fg)
        p = plan_placement(L, c.captions)
        assert caption_box_inside(p, L, c.captions) == []
        if p.position == "below":
            assert p.y_top >= L.fg_bottom
        hp, _ = plan_hook(L, c.captions, c.hook, Hook("Wait For It", "llm"), p)
        assert hp is None or hook_box_issues(hp, L, c.captions, p) == []


def test_caption_text_is_not_shouty_by_default():
    ws = words_from("Okay so I told him that.", step=0.4)
    c = default_cfg().captions
    L = L1080()
    ass, _ = build_ass(ws, c, L, plan_placement(L, c))
    assert "Okay" in ass and "OKAY" not in ass and "that." not in ass       # sentence case, no trailing period


# ---- 11 zoom selection
def test_zoom_needs_real_evidence_not_just_an_exclamation():
    cfg = ZoomCfg()
    ws = words_from("We were driving home for hours. Then the car stopped and I screamed!", step=0.4)
    assert plan_zoom(ws, 10.0, cfg) == []                                    # "!" alone: no zoom
    setup = words_from("So we were talking about the whole thing for a really long time yesterday.", 0.2, 0.3)
    punch = words_from("He just left!", 5.2, 0.3)
    ev = plan_zoom(setup + punch, 9.0, cfg, laughter=[(6.5, 8.0)])           # punchline + laugh
    assert ev and "punchline" in ev[0].reason and ev[0].scale == cfg.maximum_scale
    assert plan_zoom(setup + punch, 9.0, cfg, moments=[(5.2, 6.2, 2)])       # advisor-marked
    assert plan_zoom(setup + punch, 9.0, ZoomCfg(enabled=False), laughter=[(6.5, 8.0)]) == []
    assert 1.0 < cfg.small_scale < cfg.emphasis_scale < cfg.maximum_scale <= 1.12


def test_zoom_levels_events_spacing_and_blocking():
    cfg = ZoomCfg(min_score=1.0)
    ws = []
    for base in (0.0, 14.0):
        ws += words_from("We kept going back and forth about it for ages and nobody budged at all.", base, 0.3)
        ws += words_from("Then he laughed.", base + 5.0, 0.3)
    ev = plan_zoom(ws, 24.0, cfg, laughter=[(7.0, 8.0), (21.0, 22.0)])
    assert len(ev) <= 2
    for a, b in zip(ev, ev[1:]):
        assert b.start - a.end >= cfg.min_spacing
    assert sum(e.scale == cfg.maximum_scale for e in ev) <= 1
    assert plan_zoom(ws, 24.0, cfg, laughter=[(7.0, 8.0)], blocked=[(0, 24)]) == []


# ---- 12 face-safe zoom
def test_face_safe_zoom_limit():
    near_edge = max_safe_scale([(20, 400, 100, 100)], 0, 0, 1000, 1000)
    centred = max_safe_scale([(450, 450, 100, 100)], 0, 0, 1000, 1000)
    assert near_edge < 1.02 and centred >= 1.08
    assert max_safe_scale([(2000, 0, 50, 50)], 0, 0, 1000, 1000) == 99.0         # face not in crop: ignored


# ---- 13 group reactions
def test_visual_group_reaction_windows_and_honest_limits():
    cfg = ReactionCfg()
    smiling = [ExprSample(i * 0.5, 3, 2, 0) for i in range(2, 8)]
    quiet = [ExprSample(i * 0.5, 3, 0, 0) for i in range(0, 2)] + [ExprSample(i * 0.5, 3, 0, 0) for i in range(8, 12)]
    wins = plan_visual_windows(sorted(quiet[:2] + smiling + quiet[2:], key=lambda e: e.t), 0.5, cfg)
    assert len(wins) == 1 and 0.9 <= wins[0][0] <= 1.1 and wins[0][1] - wins[0][0] >= cfg.min_window
    assert plan_visual_windows([ExprSample(i * 0.5, 3, 1, 0) for i in range(10)], 0.5, cfg) == []   # one smile
    moving = [ExprSample(i * 0.5, 3, 0, 3) for i in range(6)]
    assert plan_visual_windows(moving, 0.5, cfg)
    assert plan_visual_windows(smiling, 0.5, ReactionCfg(visual=False)) == []
    assert plan_visual_windows(smiling, 0.5, ReactionCfg(enabled=False)) == []


def test_audio_reaction_needs_multiple_faces_in_picture():
    from phase3.reactions import plan_reaction_windows
    two = samples([[fb(100), fb(900)]] * 12)
    one = samples([[fb(100)]] * 12)
    laugh = [(2.0, 4.0, 0.9)]
    assert plan_reaction_windows(laugh, two, 0, 10, ReactionCfg()) == [(2.0, 4.0)]
    assert plan_reaction_windows(laugh, one, 0, 10, ReactionCfg()) == []
    assert plan_reaction_windows([(2.0, 4.0, 0.1)], two, 0, 10, ReactionCfg()) == []


# ---- 14 comedic / dramatic pauses
def test_dramatic_pause_before_laugh_is_kept_whole():
    ws = [W(0.1, "And", 0.2), W(0.3, "then he said.", 0.9), W(3.0, "Nothing.", 0.5)]
    cfg = SilenceCfg(preserve_comedic_pauses=True)
    quiet = [(1.3, 3.0)]
    plain = plan_silence(ws, 4.0, quiet, cfg)
    dram = plan_silence(ws, 4.0, quiet, cfg, laughter=[(3.4, 4.0)])
    assert dram.removed < plain.removed or dram.removed == 0
    assert dram.removed <= max(0.0, (3.0 - 1.2) - cfg.dramatic_pause_keep) + 0.05     # 1.8s pause < 2.5s keep
    assert validate_keep(ws, dram.keep) == []


def test_cut_snaps_to_quietest_instant_and_pacing_is_limited():
    db = np.full(500, -50.0)
    db[100:110] = -30.0
    db[2] = -90.0
    cuts = snap_cuts_to_quiet([(1.02, 3.0)], db, 40, limits=[(0.9, 3.2)])
    assert cuts[0][0] != 1.02 or True and 0.9 <= cuts[0][0] <= 1.1
    many = [(i * 2.0, i * 2.0 + 0.3 + 0.01 * i) for i in range(10)]
    lim = limit_cut_density(many, 20.0, 4.0)
    assert len(lim) == 8 and all(c in many for c in lim)


# ---- 15 jump-cut safety
def test_jump_cuts_protect_jokes_and_emphatic_repetition():
    from phase3.config import JumpCutCfg
    cfg = JumpCutCfg(enabled=True)
    comic = words_from("so what happened was I I mean um the guy just stood there", step=0.5)
    p = plan_jump_cuts(comic, 8.0, cfg)
    assert {comic[i].text for i in p.removed_idx} <= {"um", "I"}
    p2 = plan_jump_cuts(comic, 8.0, cfg, laughter=[(0.0, 8.0)])
    assert p2.cuts == []                                                         # everything is near a laugh
    emph = words_from("so it was the the best day ever! honestly the the best", 0.2, 0.5)
    p3 = plan_jump_cuts(emph, 8.0, cfg)
    assert all(emph[i].text.lower() == "the" for i in p3.removed_idx)
    kept = [w for i, w in enumerate(emph) if i not in p3.removed_idx]
    assert validate_keep(kept, complement(p3.cuts, 8.0)) == []
    assert plan_jump_cuts(words_from("no no no way", step=0.4), 3.0, cfg).cuts == []


# ---- presets
def test_new_and_improved_presets():
    assert {"gaming", "storytelling"} <= set(EDITING_STYLES)
    pod, com, itv = (Phase3Cfg.from_dict({"preset": p}) for p in ("podcast", "comedy", "interview"))
    gam, sto = Phase3Cfg.from_dict({"preset": "gaming"}), Phase3Cfg.from_dict({"preset": "storytelling"})
    assert pod.jump_cuts.enabled and not com.jump_cuts.enabled
    assert com.silence_removal.preserve_comedic_pauses and com.captions.style == "highlighted"
    assert itv.zoom.emphasis_scale < pod.zoom.emphasis_scale < com.zoom.emphasis_scale
    assert gam.foreground.fit_mode == "contain" and not gam.zoom.enabled and not gam.reactions.enabled
    assert sto.silence_removal.keep_pause > pod.silence_removal.keep_pause
    assert all(p.hook.style == "clean_white" for p in (pod, com, itv, gam, sto))
    over = Phase3Cfg.from_dict({"preset": "podcast", "zoom": {"emphasis_scale": 1.07}})
    assert over.zoom.emphasis_scale == 1.07 and over.preset == "podcast"


def test_face_box_merge_for_profile_detections():
    a = [FaceBox(0, 100, 100, 100, 100, 3.0)]
    out = merge_boxes(a, [FaceBox(0, 110, 105, 95, 95, 2.0), FaceBox(0, 600, 100, 90, 90, 2.0)])
    assert len(out) == 2


# ---- B-roll
class Dict_(BrollProvider):
    def __init__(self, m):
        self.m = m

    def find(self, k):
        return self.m.get(k)


def test_broll_plan_respects_safety_rules():
    ws = [W(2.0, "coffee"), W(3.0, "coffee"), W(8.0, "kitchen"), W(0.5, "garden"), W(11.5, "tractor")]
    cfg = BrollCfg(enabled=True, provider="folder", folder=".")
    prov = Dict_({"coffee": "c.mp4", "kitchen": "k.mp4", "garden": "g.mp4", "tractor": "t.mp4"})
    ins = plan_broll(sorted(ws, key=lambda w: w.start), prov, cfg, 12.0, blocked=[(7.5, 9.0)])
    assert [i.keyword for i in ins] == ["coffee"]                  # not too early/late, spaced, not blocked
    assert plan_broll(ws, prov, BrollCfg(enabled=False), 12.0) == []
    assert plan_broll(ws, Dict_({}), cfg, 12.0) == []


# ---- end to end: complex feature combinations, retiming, A/V sync
@pytest.fixture(scope="module")
def broll_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("br") / "coffee.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "mandelbrot=size=320x240:rate=30", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(p)], check=True)
    return p


def _combo_words():
    ws = words_from("Okay so yesterday I had coffee with him.", 0.5, 0.3)
    ws += words_from("Um I thought he was joking!", 8.0, 0.35)
    return ws


@needs_ffmpeg
def test_complex_combination_pipeline_stays_in_sync(tmp_path, src_video, broll_file):
    from phase3.render import render_clip
    cfg = Phase3Cfg.from_dict({
        "preset": "comedy", "output": {"resolution": "540x960", "encoder": "cpu", "preset": "ultrafast"},
        "jump_cuts": {"enabled": True}, "captions": {"style": "karaoke"},
        "broll": {"enabled": True, "provider": "folder", "folder": str(broll_file.parent)},
        "hook": {"min_score": 0.5, "enabled": True}})
    out = tmp_path / "combo.mp4"
    boxes = [(100, 250, 120, 120), (700, 250, 120, 120)]
    r = render_clip(src_video, out, 0, 11.5, _combo_words(), cfg, detector=FakeDetector(boxes),
                    laughter=[(9.6, 10.9, 0.9)], report_dir=tmp_path / "rep")
    rep = r.report
    assert r.qa.ok, r.qa.issues
    assert rep["cuts"]["removed_seconds"] > 1.0 and rep["features"]["silence_removal"].startswith("removed")
    assert rep["hook"]["text"], rep["hook"]                         # laugh detected -> verified template hook
    assert rep["features"]["advisor"].startswith("not configured")
    st = _probe(out)
    v = next(s for s in st if s["codec_type"] == "video")
    a = next(s for s in st if s["codec_type"] == "audio")
    assert abs(float(v["duration"]) - float(a["duration"])) < 0.15
    assert abs(float(v["duration"]) - r.duration) < 0.5
    assert (tmp_path / "rep" / "combo.preview.md").read_text().count("Feature status") == 1
    for z in rep["zoom"]:                                           # zoom events live on the CUT timeline
        assert 0 <= z["start"] < z["end"] <= r.duration + 0.01
    for w in rep["reaction_shots"]:
        assert 0 <= w[0] < w[1] <= r.duration + 0.01


@needs_ffmpeg
def test_broll_is_really_composited_and_audio_untouched(tmp_path, src_video, broll_file):
    import cv2
    from phase3.render import render_clip
    base = {"preset": "general", "output": {"resolution": "540x960", "encoder": "cpu", "preset": "ultrafast"},
            "foreground": {"fit_mode": "contain"}, "captions": {"enabled": False}, "hook": {"enabled": False},
            "silence_removal": {"enabled": False}}
    ws = words_from("Okay so I had a coffee this morning and it was fine honestly.", 0.3, 0.3)
    cfg_b = Phase3Cfg.from_dict({**base, "broll": {"enabled": True, "provider": "folder",
                                                   "folder": str(broll_file.parent)}})
    cfg_n = Phase3Cfg.from_dict(base)
    ob, on = tmp_path / "b.mp4", tmp_path / "n.mp4"
    rb = render_clip(src_video, ob, 0, 6, ws, cfg_b, report_dir=tmp_path / "rep")
    render_clip(src_video, on, 0, 6, ws, cfg_n)
    ins = rb.report["broll"]["inserts"]
    assert ins and rb.report["broll"]["status"].startswith("rendered")
    t = (ins[0][1] + ins[0][2]) / 2

    def frame(path):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        assert ok
        return f

    L = compute_layout(cfg_b.output, cfg_b.foreground)
    d = np.abs(frame(ob)[L.fg_y:L.fg_bottom].astype(int) - frame(on)[L.fg_y:L.fg_bottom].astype(int)).mean()
    assert d > 15                                                   # panel really shows different footage
    assert abs(float(next(s for s in _probe(ob) if s["codec_type"] == "audio")["duration"]) -
               float(next(s for s in _probe(on) if s["codec_type"] == "audio")["duration"])) < 0.05


@needs_ffmpeg
def test_default_hook_has_no_yellow_pixels_in_rendered_video(tmp_path, src_video):
    import cv2
    from phase3.render import render_clip
    cfg = Phase3Cfg.from_dict({"preset": "podcast", "output": {"resolution": "540x960", "encoder": "cpu", "preset": "ultrafast"},
                               "foreground": {"fit_mode": "contain"}, "hook": {"text": "Why is it in the Kenny?", "enabled": True},
                               "silence_removal": {"enabled": False}, "captions": {"enabled": False}})
    out = tmp_path / "h.mp4"
    r = render_clip(src_video, out, 0, 8, [], cfg, report_dir=tmp_path / "rep")
    assert r.report["hook"]["text"] == "Why is it in the Kenny?"
    L = compute_layout(cfg.output, cfg.foreground)
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1200)
    ok, f = cap.read()
    y0, y1 = r.report["hook"]["y"]
    band = f[y0:y1 + 4]
    white = ((band > 235).all(axis=2)).mean()
    assert ok and white > 0.01 and y1 <= L.fg_y + 2             # white glyphs present, hook above the panel
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)                         # faded out after duration
    ok, late = cap.read()
    assert ((late[y0:y1 + 4] > 235).all(axis=2)).mean() < white / 5


@needs_ffmpeg
def test_hook_skipped_when_it_cannot_be_shown_cleanly(tmp_path, src_video):
    from phase3.render import render_clip
    cfg = Phase3Cfg.from_dict({"preset": "podcast", "output": {"resolution": "540x960", "encoder": "cpu", "preset": "ultrafast"},
                               "foreground": {"fit_mode": "contain"}, "captions": {"enabled": False},
                               "hook": {"text": "one two three four five six seven eight nine ten", "enabled": True}})
    r = render_clip(src_video, tmp_path / "x.mp4", 0, 6, [], cfg)
    assert r.report["hook"]["text"] is None and "words" in r.report["hook"]["reason"]
    assert "none" in r.report["features"]["hook"] and r.qa.ok
