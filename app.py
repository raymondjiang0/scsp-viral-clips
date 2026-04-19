import os
import sys
from pathlib import Path

# Add Homebrew paths so ffmpeg/brew are found in subprocess calls
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SCSP Viral Clip Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state helpers ─────────────────────────────────────────────────────

def get_state(key, default=None):
    return st.session_state.get(key, default)


def set_state(key, value):
    st.session_state[key] = value


# ── Pipeline (defined before tabs so it's in scope when button calls it) ─────

def _run_pipeline(video_path: Path):
    from core.downloader import preprocess_for_gemini
    from core.transcriber import transcribe, format_for_prompt
    from core.analyzer import analyze_video
    from core.extractor import extract_clip, extract_thumbnail
    from rag.retriever import get_brand_context

    status = st.status("Running pipeline…", expanded=True)

    with status:
        st.write("🔧 Preprocessing video…")
        gemini_path = preprocess_for_gemini(video_path)

        st.write("🎙️ Transcribing with Whisper (this can take a few minutes)…")
        prog_t = st.progress(0.0)
        segments = transcribe(video_path, progress_callback=lambda p: prog_t.progress(min(p, 1.0)))
        transcript_text = format_for_prompt(segments)
        st.write(f"  ✅ Transcript: {len(segments)} segments")

        st.write("📚 Retrieving brand context from RAG…")
        brand_context = get_brand_context()

        st.write("🤖 Sending to Gemini 2.5 Flash for viral clip analysis…")
        prog_a = st.progress(0.0)

        def _analysis_cb(p, msg=""):
            prog_a.progress(min(p, 1.0), text=msg)

        result = analyze_video(gemini_path, brand_context, transcript_text, progress_callback=_analysis_cb)

        all_clips = result.instagram_clips + result.linkedin_clips
        st.write(f"✂️ Extracting {len(all_clips)} clips…")
        clip_paths = {}
        thumb_paths = {}
        prog_e = st.progress(0.0)

        for i, clip in enumerate(all_clips):
            try:
                cp = extract_clip(video_path, clip.start_seconds, clip.end_seconds, clip.clip_id, segments)
                clip_paths[clip.clip_id] = cp
                mid = (clip.start_seconds + clip.end_seconds) / 2
                tp = extract_thumbnail(video_path, mid, clip.clip_id)
                thumb_paths[clip.clip_id] = tp
            except Exception as e:
                st.warning(f"  ⚠️ Could not extract {clip.clip_id}: {e}")
            prog_e.progress((i + 1) / len(all_clips))

        set_state("analysis_result", result)
        set_state("clip_paths", clip_paths)
        set_state("thumb_paths", thumb_paths)
        set_state("segments", segments)
        set_state("video_path_processed", video_path)

    status.update(label="✅ Pipeline complete!", state="complete")
    st.success("Done! Switch to the Instagram Clips, LinkedIn Clips, Finished Edit, or Content Ideas tabs.")


# ── Clip card renderer (also defined before the tabs that call it) ────────────

def _render_clip_card(clip, col, tab_key: str):
    clip_paths = get_state("clip_paths", {})
    thumb_paths = get_state("thumb_paths", {})

    with col:
        with st.container(border=True):
            thumb = thumb_paths.get(clip.clip_id)
            if thumb and Path(thumb).exists():
                st.image(str(thumb), use_container_width=True)

            sc1, sc2 = st.columns(2)
            sc1.metric("Virality", f"⭐ {clip.virality_score:.1f}/10")
            sc2.metric("Duration", f"{clip.duration:.0f}s")

            st.markdown(f"**Hook:** _{clip.hook}_")

            with st.expander("Why this goes viral"):
                st.write(clip.why_viral)
                st.caption(f"Type: `{clip.content_type}`")

            with st.expander("Suggested caption + hashtags"):
                st.text_area(
                    "Caption",
                    value=clip.suggested_caption,
                    height=120,
                    key=f"caption_{clip.clip_id}_{tab_key}",
                )
                st.write(" ".join(clip.suggested_hashtags))

            start_str = f"{int(clip.start_seconds // 60):02d}:{clip.start_seconds % 60:05.2f}"
            end_str = f"{int(clip.end_seconds // 60):02d}:{clip.end_seconds % 60:05.2f}"
            st.caption(f"⏱ {start_str} → {end_str}")

            clip_file = clip_paths.get(clip.clip_id)
            if clip_file and Path(clip_file).exists():
                with open(clip_file, "rb") as f:
                    st.download_button(
                        "⬇ Download clip",
                        data=f,
                        file_name=f"{clip.clip_id}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_{clip.clip_id}_{tab_key}",
                    )
                if st.button("▶ Preview", key=f"prev_{clip.clip_id}_{tab_key}", use_container_width=True):
                    set_state(f"preview_{tab_key}", clip.clip_id)

            if st.button("👍 Mark as high-performer", key=f"fb_{clip.clip_id}_{tab_key}", use_container_width=True):
                try:
                    from rag.indexer import add_performing_clip
                    add_performing_clip(
                        f"{clip.hook} — {clip.content_type} clip on {clip.platform}",
                        clip.why_viral,
                    )
                    st.success("Saved to knowledge base!")
                except Exception as e:
                    st.warning(f"Could not save: {e}")

            selected = get_state("selected_clips", [])
            is_selected = clip.clip_id in selected
            label = "✅ In final edit" if is_selected else "Add to final edit"
            if st.button(label, key=f"sel_{clip.clip_id}_{tab_key}", use_container_width=True):
                if is_selected:
                    selected.remove(clip.clip_id)
                else:
                    selected.append(clip.clip_id)
                set_state("selected_clips", selected)
                st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 SCSP Clip Engine")
    st.caption("Viral short-form content from long-form video")
    st.divider()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        st.success("Gemini API key loaded", icon="✅")
    else:
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            st.rerun()

    st.divider()
    st.subheader("Knowledge Base")

    try:
        from rag.indexer import is_indexed
        indexed = is_indexed()
        if indexed:
            st.success("Brand guide + SCSP site indexed", icon="📚")
        else:
            st.warning("Not indexed yet — click below", icon="⚠️")
    except Exception:
        indexed = False
        st.warning("RAG unavailable (run pip install first)", icon="⚠️")

    if st.button("(Re)index Brand Guide + Website", use_container_width=True):
        try:
            from rag.indexer import index_all
            status_box = st.empty()
            with st.spinner("Indexing…"):
                n = index_all(status_callback=lambda msg: status_box.caption(msg))
            st.success(f"Indexed {n} chunks!")
            st.rerun()
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    with st.expander("Mark a clip as high-performing"):
        fb_desc = st.text_area("Clip description", placeholder="What the clip was about…", key="fb_desc")
        fb_why = st.text_area("Why it worked", placeholder="What made it go viral…", key="fb_why")
        if st.button("Save to knowledge base", key="fb_save"):
            if fb_desc and fb_why:
                try:
                    from rag.indexer import add_performing_clip
                    add_performing_clip(fb_desc, fb_why)
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    st.divider()
    st.caption(f"Model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    st.caption(f"Whisper: {os.getenv('WHISPER_MODEL', 'large-v3')}")


# ── Gate on API key ───────────────────────────────────────────────────────────

if not os.getenv("GEMINI_API_KEY"):
    st.error("Add your Gemini API key in the sidebar or in your `.env` file to get started.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_process, tab_ig, tab_li, tab_edit, tab_ideas = st.tabs([
    "⚙️ Process Video",
    "📱 Instagram Clips",
    "💼 LinkedIn Clips",
    "🎬 Finished Edit",
    "💡 Content Ideas",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Process Video
# ─────────────────────────────────────────────────────────────────────────────

with tab_process:
    st.header("Process a Video")

    from core.downloader import check_ffmpeg
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        st.error("ffmpeg not found. Run: `brew install ffmpeg` then restart the app.")

    input_method = st.radio(
        "Input source",
        ["YouTube URL", "Upload MP4 / MOV"],
        horizontal=True,
    )

    video_path = get_state("video_path")

    if input_method == "YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if url and st.button("Download Video", use_container_width=True, disabled=not ffmpeg_ok):
            from core.downloader import download_youtube
            prog = st.progress(0.0, text="Downloading…")
            try:
                vp = download_youtube(url, progress_callback=lambda p: prog.progress(p, text=f"Downloading… {p:.0%}"))
                set_state("video_path", vp)
                prog.progress(1.0, text="Downloaded!")
                st.success(f"Saved: `{vp.name}`")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")
    else:
        uploaded = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
        if uploaded and st.button("Use this file", use_container_width=True):
            from core.downloader import save_upload
            vp = save_upload(uploaded)
            set_state("video_path", vp)
            st.success(f"Saved: `{vp.name}`")
            st.rerun()

    if video_path and Path(video_path).exists():
        st.info(f"Ready to process: **{video_path.name}**")

        from core.downloader import get_video_duration
        duration_s = get_video_duration(video_path)
        minutes = int(duration_s // 60)
        est_cost = duration_s * 0.0001
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{minutes}m {int(duration_s % 60)}s")
        col2.metric("Est. Gemini cost", f"~${est_cost:.3f}")
        col3.metric("File size", f"{Path(video_path).stat().st_size / 1e6:.1f} MB")

        st.divider()

        if st.button("🚀 Analyze for Viral Clips", type="primary", use_container_width=True, disabled=not ffmpeg_ok):
            _run_pipeline(Path(video_path))
    else:
        st.caption("No video loaded yet.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Instagram Clips
# ─────────────────────────────────────────────────────────────────────────────

with tab_ig:
    result = get_state("analysis_result")
    if not result:
        st.info("Process a video first (⚙️ Process Video tab).")
    else:
        st.header(f"📱 Instagram Clips — {len(result.instagram_clips)} suggestions")

        sort_by = st.selectbox("Sort by", ["Virality score", "Duration", "Clip order"], key="ig_sort")
        clips = list(result.instagram_clips)
        if sort_by == "Virality score":
            clips.sort(key=lambda c: c.virality_score, reverse=True)
        elif sort_by == "Duration":
            clips.sort(key=lambda c: c.duration)

        preview_id = get_state("preview_ig")
        if preview_id:
            cp = get_state("clip_paths", {}).get(preview_id)
            if cp and Path(cp).exists():
                st.video(str(cp))
            if st.button("Close preview", key="close_ig_prev"):
                set_state("preview_ig", None)
                st.rerun()

        cols = st.columns(2)
        for i, clip in enumerate(clips):
            _render_clip_card(clip, cols[i % 2], "ig")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: LinkedIn Clips
# ─────────────────────────────────────────────────────────────────────────────

with tab_li:
    result = get_state("analysis_result")
    if not result:
        st.info("Process a video first (⚙️ Process Video tab).")
    else:
        st.header(f"💼 LinkedIn Clips — {len(result.linkedin_clips)} suggestions")

        sort_by_li = st.selectbox("Sort by", ["Virality score", "Duration", "Clip order"], key="li_sort")
        clips_li = list(result.linkedin_clips)
        if sort_by_li == "Virality score":
            clips_li.sort(key=lambda c: c.virality_score, reverse=True)
        elif sort_by_li == "Duration":
            clips_li.sort(key=lambda c: c.duration)

        preview_id_li = get_state("preview_li")
        if preview_id_li:
            cp = get_state("clip_paths", {}).get(preview_id_li)
            if cp and Path(cp).exists():
                st.video(str(cp))
            if st.button("Close preview", key="close_li_prev"):
                set_state("preview_li", None)
                st.rerun()

        cols_li = st.columns(2)
        for i, clip in enumerate(clips_li):
            _render_clip_card(clip, cols_li[i % 2], "li")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Finished Edit
# ─────────────────────────────────────────────────────────────────────────────

with tab_edit:
    result = get_state("analysis_result")
    if not result:
        st.info("Process a video first (⚙️ Process Video tab).")
    else:
        st.header("🎬 Finished Edit")

        fp = result.finished_product
        st.info(
            f"**Suggested narrative:** {fp.narrative_arc}\n\n"
            f"**Hook text:** _{fp.suggested_hook_text}_\n\n"
            f"**CTA:** _{fp.suggested_cta}_"
        )

        selected = get_state("selected_clips", [])
        all_clips_map = {c.clip_id: c for c in result.instagram_clips + result.linkedin_clips}
        clip_paths = get_state("clip_paths", {})

        if not selected and fp.clips_in_order:
            selected = [c for c in fp.clips_in_order if c in clip_paths]
            set_state("selected_clips", selected)

        st.subheader(f"Selected clips ({len(selected)})")
        st.caption("Add/remove clips from the Instagram or LinkedIn tabs.")

        if not selected:
            st.warning("No clips selected yet.")
        else:
            for i, cid in enumerate(selected):
                clip = all_clips_map.get(cid)
                if not clip:
                    continue
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{i+1}.** `{cid}` — {clip.hook[:80]}…")
                c2.write(f"⭐ {clip.virality_score} | {clip.duration:.0f}s")

            st.divider()

            add_hook = st.checkbox("Add hook text overlay (first 2 seconds)", value=True)
            hook_text = st.text_input("Hook overlay text", value=fp.suggested_hook_text, disabled=not add_hook)

            if st.button("🎬 Generate Final Video", type="primary", use_container_width=True):
                with st.spinner("Assembling final video…"):
                    try:
                        from core.editor import assemble_clips, add_text_overlay
                        ordered_paths = [Path(clip_paths[cid]) for cid in selected if cid in clip_paths]
                        valid_paths = [p for p in ordered_paths if p.exists()]
                        if not valid_paths:
                            st.error("No valid clip files found. Re-run processing.")
                        else:
                            assembled = assemble_clips(valid_paths, "final_edit.mp4")
                            if add_hook and hook_text:
                                assembled = add_text_overlay(assembled, hook_text)
                            set_state("final_video_path", assembled)
                            st.success("Final video ready!")
                    except Exception as e:
                        st.error(f"Assembly failed: {e}")

            final = get_state("final_video_path")
            if final and Path(final).exists():
                st.video(str(final))
                with open(final, "rb") as f:
                    st.download_button(
                        "⬇ Download Final Video",
                        data=f,
                        file_name="scsp_viral_edit.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Content Ideas
# ─────────────────────────────────────────────────────────────────────────────

with tab_ideas:
    result = get_state("analysis_result")
    if not result:
        st.info("Process a video first (⚙️ Process Video tab).")
    else:
        st.header("💡 Original Content Ideas")
        st.caption(
            "Based on this video's content + SCSP brand knowledge — original concepts to film, "
            "separate from the clips above."
        )

        for i, idea in enumerate(result.content_suggestions, 1):
            with st.expander(f"{i}. {idea.title}", expanded=(i == 1)):
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.markdown(f"**Hook:** _{idea.hook}_")
                    st.markdown("**Outline:**")
                    for pt in idea.outline:
                        st.markdown(f"- {pt}")
                    st.markdown(f"**Why it works:** {idea.why_viral}")
                    st.markdown(f"**Trending angle:** {idea.trending_angle}")

                with col_b:
                    platform_color = {"instagram": "🟣", "linkedin": "🔵", "both": "⚡"}.get(
                        idea.target_platform, "⚡"
                    )
                    st.metric("Platform", f"{platform_color} {idea.target_platform.title()}")
                    st.metric("Format", idea.format.replace("_", " ").title())

                if st.button(f"Save idea #{i} to content calendar", key=f"save_idea_{i}"):
                    from config import OUTPUT_DIR
                    cal = OUTPUT_DIR / "content_calendar.txt"
                    with open(cal, "a") as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"TITLE: {idea.title}\nPLATFORM: {idea.target_platform}\nFORMAT: {idea.format}\n")
                        f.write(f"HOOK: {idea.hook}\nOUTLINE:\n")
                        for pt in idea.outline:
                            f.write(f"  - {pt}\n")
                        f.write(f"WHY VIRAL: {idea.why_viral}\nTRENDING ANGLE: {idea.trending_angle}\n")
                    st.success(f"Saved to `outputs/content_calendar.txt`")
