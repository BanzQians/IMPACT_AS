#!/usr/bin/env python3
"""Generate IMPACT-Scribe User Study Guide PDF with embedded screenshots."""

from pathlib import Path
from fpdf import FPDF

ASSETS = Path(__file__).resolve().parent / "assets" / "quick_start"
OUT = Path(__file__).resolve().parent / "IMPACT_Scribe_User_Study_Guide.pdf"

# Use DejaVu TTF fonts for proper Unicode rendering
DEJAVU_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = DEJAVU_DIR / "DejaVuSans.ttf"
FONT_BOLD = DEJAVU_DIR / "DejaVuSans-Bold.ttf"


class GuidePDF(FPDF):
    MARGIN = 18
    COL_W = 174  # A4 width(210) - 2*MARGIN

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_left_margin(self.MARGIN)
        self.set_right_margin(self.MARGIN)

        if FONT_REGULAR.exists() and FONT_BOLD.exists():
            self.add_font("dv", "", str(FONT_REGULAR))
            self.add_font("dv", "B", str(FONT_BOLD))
            self._fn = "dv"
        else:
            self._fn = "Helvetica"

    # ---- helpers ----
    def _font(self, style="", size=11):
        self.set_font(self._fn, style, size)

    def _color(self, r, g, b):
        self.set_text_color(r, g, b)

    def _title_page(self):
        self.add_page()
        self.ln(50)
        self._font("B", 28)
        self._color(23, 37, 52)
        self.cell(0, 14, "IMPACT-Scribe", ln=True, align="C")
        self._font("B", 20)
        self._color(80, 100, 120)
        self.cell(0, 12, "User Study Guide", ln=True, align="C")
        self.ln(10)
        self._font("", 12)
        self._color(100, 110, 120)
        self.cell(0, 8, "Interactive Action Segmentation Review Tool", ln=True, align="C")
        self.ln(30)
        self._font("", 11)
        self._color(60, 70, 80)
        self.cell(0, 7, "CVHCI  |  Karlsruhe Institute of Technology", ln=True, align="C")
        self.cell(0, 7, "Version 1.0  |  April 2026", ln=True, align="C")

    def _section(self, title, level=1):
        self.ln(4 if level == 1 else 2)
        if level == 1:
            self._font("B", 16)
            self._color(23, 37, 52)
        else:
            self._font("B", 13)
            self._color(50, 65, 80)
        self.cell(0, 9, title, ln=True)
        if level == 1:
            self.set_draw_color(200, 210, 220)
            self.line(self.MARGIN, self.get_y(), self.MARGIN + self.COL_W, self.get_y())
            self.ln(3)
        else:
            self.ln(1)

    def _para(self, text):
        self._font("", 11)
        self._color(40, 50, 60)
        self.multi_cell(self.COL_W, 6, text)
        self.ln(2)

    def _bullet(self, text, indent=6):
        self._font("", 11)
        self._color(40, 50, 60)
        self.set_x(self.MARGIN + indent)
        self.cell(4, 6, chr(8226))
        self.multi_cell(self.COL_W - indent - 4, 6, text)

    def _bold_bullet(self, bold_part, rest):
        self._font("", 11)
        self._color(40, 50, 60)
        self.set_x(self.MARGIN + 6)
        self.cell(4, 6, chr(8226))
        self._font("B", 11)
        bw = self.get_string_width(bold_part) + 1
        self.cell(bw, 6, bold_part)
        self._font("", 11)
        remaining = self.COL_W - 10 - bw
        if remaining < 30:
            self.ln(6)
            self.set_x(self.MARGIN + 10)
            remaining = self.COL_W - 10
        self.multi_cell(remaining, 6, rest)

    def _img(self, filename, caption=""):
        img_path = ASSETS / filename
        if not img_path.exists():
            self._para(f"[Image not found: {filename}]")
            return
        avail_w = self.COL_W
        self.image(str(img_path), x=self.MARGIN, w=avail_w)
        if caption:
            self._font("", 9)
            self._color(120, 130, 140)
            self.cell(0, 5, caption, ln=True, align="C")
        self.ln(3)

    def _shortcut_row(self, key, desc):
        self._font("B", 10)
        self._color(50, 60, 80)
        self.cell(35, 6, key)
        self._font("", 10)
        self._color(40, 50, 60)
        self.cell(0, 6, desc, ln=True)

    # ---- content ----
    def build(self):
        self._title_page()

        # ======== 1. Overview ========
        self.add_page()
        self._section("1. Overview")
        self._para(
            "IMPACT-Scribe is an interactive action segmentation review tool. "
            "Given a video and an initial (machine-generated) action segmentation, "
            "you will review and correct the segmentation by:"
        )
        self._bullet("Drawing segments directly on the timeline (drag to create, resize, move, or delete)")
        self._bullet("Following the system's query suggestions (boundary or label questions)")
        self._bullet("Drawing temporal scribbles to refine boundaries")
        self._bullet("Accepting or rejecting proposed corrections")
        self.ln(2)
        self._para(
            "Your goal is to produce a high-quality action segmentation with minimal effort. "
            "The tool learns from each of your corrections: every Accept or Reject updates "
            "the internal model so that subsequent suggestions become more accurate."
        )

        # ======== 2. Getting Started ========
        self._section("2. Getting Started")

        self._section("Step 1: Load a Video or Baseline", level=2)
        self._para(
            'Click the menu button in the top-left toolbar and choose "Open Session". '
            "Select a video file and, optionally, an existing annotation JSON. "
            "The system loads the video and displays the baseline segmentation on the timeline."
        )
        self._img("step_01_load_baseline.png", "Fig 1. Load a video and baseline segmentation")

        self._section("Step 2: Inspect the Workspace", level=2)
        self._para(
            "After loading, the workspace shows the video player on the left and "
            "the action timeline on the right. Each colored segment represents one "
            "action label. Hover over the timeline to preview any frame."
        )
        self._img("step_02_loaded_workspace.png", "Fig 2. Workspace with video and timeline")

        self._section("Step 3: Click Suggest Query", level=2)
        self._para(
            'Click the "Suggest Query" button in the toolbar. The system analyzes '
            "the current segmentation and suggests the most valuable boundary or "
            "label to review next. A suggestion card appears in the footer area."
        )
        self._img("step_03_suggest_query.png", "Fig 3. Suggest Query generates a review target")

        self._section("Step 4: Review the Suggestion", level=2)
        self._para(
            "The footer card shows the suggested question: a boundary to check or "
            "a label to verify. Read the description and decide whether to accept, "
            "reject, or refine with a scribble."
        )
        self._img("step_04_review_suggestion.png", "Fig 4. Review the suggested boundary or label")

        self._section("Step 5: Refine and Accept", level=2)
        self._para(
            "If the boundary needs adjustment, draw an uncertain scribble stroke "
            "across the suspicious region on the timeline. The system proposes "
            "a refined boundary (red line). Drag to fine-tune, then click Accept."
        )
        self._img("step_05_refine_and_accept.png", "Fig 5. Draw a scribble, refine, and accept")

        # ======== 3. Core Workflows ========
        self.add_page()
        self._section("3. Core Workflows")

        self._section("3.1 Direct Timeline Annotation (Basic)", level=2)
        self._para(
            "The simplest way to annotate is to draw segments directly on the "
            "timeline. No special mode is required - this works at any time:"
        )
        self._bullet("1. Select a label from the Label Panel on the left (click a Verb, then an Object)")
        self._bullet(
            "2. On the timeline, click and drag in an empty area to draw a new segment. "
            "The segment appears as a colored block matching the selected label."
        )
        self._bullet(
            "3. To resize: drag the left or right edge of any existing segment"
        )
        self._bullet(
            "4. To move: drag the center of an existing segment"
        )
        self._bullet(
            "5. To delete: right-click on a segment to remove it"
        )
        self._bullet(
            "6. Ctrl+Click on a segment to split it at that frame"
        )
        self.ln(2)
        self._para(
            "This is useful when you want to annotate from scratch or quickly "
            "fix a region without using the query planner."
        )

        self._section("3.2 Query Suggestion Workflow (Guided)", level=2)
        self._para(
            "For efficient review, the query planner helps you focus on the "
            "most valuable corrections. It selects the next question based on:"
        )
        self._bullet("Boundary uncertainty: where the model is least confident")
        self._bullet("Label disagreement: where features suggest a different label")
        self._bullet("Multi-view conflict: where different camera views disagree")
        self._bullet("State conflict: where action labels violate assembly rules")
        self.ln(2)
        self._para("After clicking Suggest Query, three actions are available:")
        self._bold_bullet("Accept Suggestion ", "- Apply the proposed change directly")
        self._bold_bullet("Start Scribble ", "- Enter scribble mode to refine the boundary first")
        self._bold_bullet("Reject Suggestion ", "- Skip this query and move on")
        self.ln(2)
        self._para("Repeat: click Suggest Query again after each decision. "
                    "The planner adapts based on your corrections.")

        self._section("3.3 Temporal Scribble Interaction", level=2)
        self._para(
            "Scribbles are short strokes you draw on the timeline in Boundary Scribble mode "
            "to tell the system where a boundary might be. There are three scribble types:"
        )
        self._bold_bullet("Uncertain (default) ", '- "I think a boundary is somewhere in this range"')
        self._bold_bullet("Left ", '- "The left side of this region belongs to this label"')
        self._bold_bullet("Right ", '- "The right side of this region belongs to this label"')
        self.ln(2)
        self._para("Scribble workflow:")
        self._bullet('1. Enter scribble mode via the Interaction dropdown or "Start Scribble" button')
        self._bullet("2. Click and drag on the timeline across the suspicious boundary region")
        self._bullet("3. The system proposes a boundary split (red marker) with left/right labels")
        self._bullet("4. Optionally drag the red marker to fine-tune the exact position")
        self._bullet("5. Click Accept to apply, or Reject to discard")
        self._bullet('6. Click "Clear Scribbles" to reset the current scribble session')

        self._section("3.4 Manual Global Segmentation", level=2)
        self._para(
            'Select "Manual Segmentation" from the Interaction dropdown. This activates '
            "a full manual mode where segment boundaries are placed frame by frame. "
            "Use this for precise control when the automated workflow is not needed."
        )

        # ======== 4. Interface Reference ========
        self.add_page()
        self._section("4. Interface Reference")

        self._section("4.1 Toolbar Buttons", level=2)
        self._bold_bullet("Play / Pause ", "- Toggle video playback")
        self._bold_bullet("<< / >> ", "- Step backward / forward by 10 frames")
        self._bold_bullet("ASOT Pre-label ", "- Generate a machine baseline segmentation")
        self._bold_bullet("Magnifier ", "- Toggle zoom selection on the video")
        self._bold_bullet("Validation ", "- Toggle validation overlay on/off")
        self._bold_bullet("Interaction dropdown ", "- Switch between Boundary Scribble / Manual Segmentation / Exit")
        self._bold_bullet("Clear Scribbles ", "- Reset current scribble strokes and proposals")
        self._bold_bullet("Suggest Query ", "- Ask the planner for the next review target")
        self._bold_bullet("Settings ", "- Open settings dialog (Ctrl+,)")
        self._bold_bullet("Quick Start ", "- Open the in-app quick start guide")
        self._bold_bullet("+ Add View ", "- Add an additional camera view")

        self._section("4.2 Timeline", level=2)
        self._para("The timeline is the primary annotation area:")
        self._bullet("Colored blocks = action segments with label names")
        self._bullet("Red vertical line = current playhead position")
        self._bullet("Hover = preview that frame in the video player")
        self._bullet("Dotted lines = segment boundaries (draggable)")
        self._bullet("Red marker = proposed boundary from scribble refinement")
        self._bullet("Scroll wheel = pan the timeline left/right")
        self.ln(2)
        self._para("Mouse actions on the timeline:")
        self._bullet("Left click + drag on empty space = create a new segment (uses selected label)")
        self._bullet("Left drag on segment edge = resize segment boundary")
        self._bullet("Left drag on segment center = move the entire segment")
        self._bullet("Right-click on segment = delete it")
        self._bullet("Ctrl + click on segment = split at that frame")

        self._section("4.3 Label Panel", level=2)
        self._para(
            "The label panel (left side) shows all available action labels organized "
            "by verb-object structure. Click a verb to filter, then click an object to "
            "select the label for annotation."
        )
        self._bullet("Search box: filter labels by name")
        self._bullet("Add button: create a new label with name, ID, and color")
        self._bullet("Double-click: rename an existing label inline")
        self._bullet('Escape labels: "Unknown", "Other", and "Background" are always available '
                     "at the top of the panel. Use them to mark segments you are unsure about.")

        self._section("4.4 Footer / Query Card", level=2)
        self._para("When a query is active, the footer shows:")
        self._bullet("Query type: BOUNDARY SUGGESTION or LABEL SUGGESTION")
        self._bullet("Target frame range and labels involved")
        self._bullet("Confidence score")
        self._bullet("Action buttons: Accept / Reject / Start Scribble")

        # ======== 5. Keyboard Shortcuts ========
        self.add_page()
        self._section("5. Keyboard Shortcuts")

        self._section("Navigation", level=2)
        self._shortcut_row("Space", "Play / Pause")
        self._shortcut_row("A", "Step back 1 frame")
        self._shortcut_row("D", "Step forward 1 frame")
        self._shortcut_row("Shift+A", "Step back 10 frames")
        self._shortcut_row("Shift+D", "Step forward 10 frames")
        self._shortcut_row("J", "Seek back 1 second")
        self._shortcut_row("K", "Pause")
        self._shortcut_row("L", "Seek forward 1 second")
        self._shortcut_row("Home", "Jump to start")
        self._shortcut_row("End", "Jump to end")
        self.ln(3)

        self._section("Editing", level=2)
        self._shortcut_row("Ctrl+Z", "Undo")
        self._shortcut_row("Ctrl+Y", "Redo")
        self._shortcut_row("Ctrl+,", "Open Settings")
        self.ln(3)

        self._section("Assembly State Mode (PSR)", level=2)
        self._shortcut_row("Ctrl+K", "Split segment at playhead")
        self._shortcut_row("Ctrl+Shift+S", "Set scope to current segment")
        self._shortcut_row("Ctrl+Shift+F", "Set scope from current frame forward")
        self._shortcut_row("Ctrl+Backspace", "Reset selected segment")
        self._shortcut_row("Ctrl+I", "Invert selected segment state")
        self._shortcut_row("Ctrl+M", "Merge adjacent identical states")
        self.ln(3)

        self._section("Video Player", level=2)
        self._shortcut_row("Ctrl+Scroll", "Zoom in / out on video")
        self._shortcut_row("Left Drag", "Pan zoomed video view")
        self._shortcut_row("Double Click", "Reset zoom to fit window")

        # ======== 6. Multi-View ========
        self._section("6. Multi-View Support")
        self._para(
            "IMPACT-Scribe supports up to 5 synchronized camera views. "
            'Click "+ Add View" to load an additional video. All views share '
            "the same timeline and are synchronized frame-by-frame."
        )
        self._bullet("Hovering or seeking in one view updates all views")
        self._bullet("Labels can differ between views (per-view annotation)")
        self._bullet("The query planner considers cross-view disagreement")

        # ======== 7. Import / Export ========
        self._section("7. Import / Export")
        self._bold_bullet("Open Session ", "- Load video + optional label map + annotation JSON")
        self._bold_bullet("Import JSON ", "- Load annotation data for selected views")
        self._bold_bullet("Export JSON ", "- Save all annotations to a single JSON file")
        self._bold_bullet("Export per-view ", "- Save separate JSON files per camera view")
        self._bold_bullet("Import/Export Label Map ", "- Load or save the label vocabulary (TXT)")

        # ======== 8. User Study Task ========
        self.add_page()
        self._section("8. User Study Task Instructions")

        self._section("8.1 Your Task", level=2)
        self._para(
            "You will be given a video with a machine-generated action segmentation baseline. "
            "Your task is to review and correct this segmentation using the tool's guided workflow."
        )

        self._section("8.2 Procedure", level=2)
        self._bullet("1. The system will load a pre-configured session for you")
        self._bullet('2. Click "Suggest Query" to receive the first review question')
        self._bullet("3. For each suggestion, decide: Accept, Reject, or Refine with Scribble")
        self._bullet('4. After each decision, click "Suggest Query" again')
        self._bullet("5. Continue until you are satisfied with the segmentation quality")
        self._bullet("6. Export the final annotation when done")
        self.ln(2)

        self._section("8.3 Tips", level=2)
        self._bullet("You can always draw segments directly on the timeline without any special mode")
        self._bullet("Trust the query planner: it prioritizes the most impactful corrections first")
        self._bullet("Use scribbles when a boundary is close but not exact")
        self._bullet("Use Accept when the suggestion looks correct")
        self._bullet("Use Reject to skip suggestions you disagree with")
        self._bullet('Use "Unknown" or "Other" labels for segments you cannot identify confidently')
        self._bullet("Hover over the timeline to quickly preview frames before editing")
        self._bullet("Use A/D keys for precise frame-by-frame inspection")
        self._bullet("Ctrl+Z to undo any mistake")
        self.ln(2)

        self._section("8.4 What Counts as Done", level=2)
        self._para(
            "There is no fixed number of corrections required. You are done when "
            "you believe the segmentation accurately reflects the actions in the video. "
            "The system will record all your interactions for analysis."
        )

        # ======== 9. Troubleshooting ========
        self._section("9. Troubleshooting")
        self._bold_bullet("No suggestion available: ", "Make sure a video and baseline are loaded first.")
        self._bold_bullet("Scribble not working: ", 'Ensure you are in "Boundary Scribble" mode from the Interaction dropdown.')
        self._bold_bullet("Video not playing: ", "Check that the video file path is accessible. Try re-loading.")
        self._bold_bullet("Labels missing: ", "Import a label map TXT file via the menu.")
        self._bold_bullet("Need help: ", "Click the Quick Start button (top-right) for the in-app guide.")

        self.output(str(OUT))
        return OUT


if __name__ == "__main__":
    pdf = GuidePDF()
    pdf.build()
    print(f"Generated: {OUT}")
