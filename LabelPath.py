"""
Interactive path labeling tool — manually walk from start to end using arrow keys,
recording the labeled path step by step.

Usage:
  python LabelPath.py                     # start from index 0, label one by one
  python LabelPath.py --index 5           # label a single trajectory at index 5
  python LabelPath.py --batch             # label all trajectories in sequence
  python LabelPath.py --output my_labels  # custom output directory
"""

import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils.tools import mapdata_to_modelmatrix, calculate_match_rate

# ========================== Constants ==========================

DX_DY = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
#         right     up        left       down

MAP_ROWS = 529
MAP_COLS = 564

MODE_COLORS = {
    "TG": "orange",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}
MODE_LIST = ["GSD", "GG", "TS", "TG"]

DEFAULT_MAPDATA_PATH = "data/GridModesAdjacentRealworld.pkl"
DEFAULT_CSV_PATH = "data/data_lower_test.csv"
DEFAULT_OUTPUT_DIR = "label_output"

DISTANCE_THRESHOLD = 1.0
VIEW_PADDING = 30

# Label options after saving (press 1-5 to select)
LABEL_OPTIONS = {
    "1": "Expressway",
    "2": "Nat/Prov Road",
    "3": "HS Railway",
    "4": "Conv Railway",
    "5": "Other",
}

# ========================== Key Bindings ==========================
KEY_MAP = {
    "right": ("move", 0),
    "up":    ("move", 1),
    "left":  ("move", 2),
    "down":  ("move", 3),
    "d": ("move", 0),
    "w": ("move", 1),
    "a": ("move", 2),
    "s": ("move", 3),
    "backspace": ("undo", None),
    "ctrl+z":    ("undo", None),
    "r":         ("reset", None),
    "enter":     ("save", None),
    # label keys: handled separately in on_key during label-selection mode
}

# ========================== Data Loading ==========================

def load_mapdata(path=DEFAULT_MAPDATA_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_traj_csv(path=DEFAULT_CSV_PATH):
    return pd.read_csv(path)


def build_multi_mapdata(raw_mapdata, selected_mode):
    """Build multi_mapdata for the given mode(s), matching PathEnv.
    Returns ndarray of shape (564, 529).
    """
    mode_matrices = mapdata_to_modelmatrix(raw_mapdata, MAP_ROWS, MAP_COLS)
    if isinstance(selected_mode, str):
        selected_mode = [selected_mode]
    multi = np.zeros((MAP_COLS, MAP_ROWS), dtype=np.int32)
    for m in selected_mode:
        if m in mode_matrices:
            multi += np.array(mode_matrices[m], dtype=np.int32)
    return np.clip(multi, 0, 1)


# ========================== State ==========================

class LabelState:
    """Holds labeling state for a single trajectory."""

    def __init__(self, row, multi_mapdata):
        self.row = row
        self.order = int(row["order"]) if "order" in row.index else 0
        self.mode = str(row["mode"]).strip()
        self.start_x = float(row["locx_o"])
        self.start_y = float(row["locy_o"])
        self.end_x = float(row["locx_d"])
        self.end_y = float(row["locy_d"])

        self.multi_mapdata = multi_mapdata

        self.cur_x = int(round(self.start_x))
        self.cur_y = int(round(self.start_y))

        self.path_history = [(self.cur_x, self.cur_y)]
        self.step_count = 0

    @property
    def reached(self):
        dist = abs(self.cur_x - self.end_x) + abs(self.cur_y - self.end_y)
        return dist <= DISTANCE_THRESHOLD

    @property
    def remaining_dist(self):
        return abs(self.cur_x - self.end_x) + abs(self.cur_y - self.end_y)

    def current_match_rate(self):
        if len(self.path_history) <= 1:
            return 0.0
        return calculate_match_rate(self.path_history, self.multi_mapdata)

    def can_move(self, dx, dy):
        nx = self.cur_x + dx
        ny = self.cur_y + dy
        return 0 <= nx < MAP_COLS and 0 <= ny < MAP_ROWS

    def apply_move(self, action_id):
        dx, dy = DX_DY[action_id]
        self.cur_x += dx
        self.cur_y += dy
        self.path_history.append((self.cur_x, self.cur_y))
        self.step_count += 1

    def undo(self):
        if len(self.path_history) > 1:
            self.path_history.pop()
            self.cur_x, self.cur_y = self.path_history[-1]
            self.step_count = max(0, self.step_count - 1)
            return True
        return False

    def reset(self):
        self.cur_x = int(round(self.start_x))
        self.cur_y = int(round(self.start_y))
        self.path_history = [(self.cur_x, self.cur_y)]
        self.step_count = 0


# ========================== Renderer ==========================

class PathRenderer:
    """Manages the matplotlib figure and incremental updates."""

    def __init__(self, state: LabelState, raw_mapdata: dict):
        self.state = state

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("LabelPath — Interactive Path Labeling")

        # ---- road network: continuous RGB image, one color per mode ----
        road_rgb = self._build_road_rgb(raw_mapdata)  # shape (529, 564, 3)
        self.ax.imshow(
            road_rgb,
            extent=[0, MAP_COLS, 0, MAP_ROWS],
            origin="lower",
            alpha=0.45,
            zorder=1,
        )

        # ---- start / end markers ----
        self.start_handle = self.ax.scatter(
            state.start_x, state.start_y,
            c="limegreen", marker="o", s=180,
            edgecolors="darkgreen", linewidths=2,
            zorder=5, label="Start",
        )
        self.end_handle = self.ax.scatter(
            state.end_x, state.end_y,
            c="red", marker="X", s=180,
            edgecolors="darkred", linewidths=2,
            zorder=5, label="End",
        )

        # ---- path trail (initially empty) ----
        (self.path_line,) = self.ax.plot(
            [], [], "-",
            color="crimson", linewidth=2.5, alpha=0.85,
            zorder=3, label="Path",
        )

        # ---- cursor (small ring so roads are visible underneath) ----
        self.cursor = self.ax.scatter(
            state.cur_x, state.cur_y,
            c="cyan", marker="o", s=25,
            edgecolors="darkblue", linewidths=1.5,
            zorder=6, label="Cursor",
        )

        # ---- viewport crop around start-end ----
        x_vals = [state.start_x, state.end_x]
        y_vals = [state.start_y, state.end_y]
        x_center = (min(x_vals) + max(x_vals)) / 2
        y_center = (min(y_vals) + max(y_vals)) / 2
        x_range = max(abs(max(x_vals) - min(x_vals)), 20)
        y_range = max(abs(max(y_vals) - min(y_vals)), 20)

        self.ax.set_xlim(
            max(0, x_center - x_range / 2 - VIEW_PADDING),
            min(MAP_COLS, x_center + x_range / 2 + VIEW_PADDING),
        )
        self.ax.set_ylim(
            max(0, y_center - y_range / 2 - VIEW_PADDING),
            min(MAP_ROWS, y_center + y_range / 2 + VIEW_PADDING),
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)

        # ---- mode color legend ----
        mode_handles = [
            Line2D([0], [0], color="orange", lw=3, label="TG"),
            Line2D([0], [0], color="blue", lw=3, label="GG"),
            Line2D([0], [0], color="green", lw=3, label="GSD"),
            Line2D([0], [0], color="red", lw=3, label="TS"),
        ]
        # Merge with existing handles (start/end/path/cursor)
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles=mode_handles + handles, loc="upper right")

        self._update_title()
        self._draw_legend_box()

        self.fig.tight_layout()

    def _build_road_rgb(self, raw_mapdata):
        """Build an RGB image (529, 564, 3) from all 4 mode grids.
        Each mode contributes its color; overlapping pixels are blended.
        Background = white.  Transposed for imshow: horizontal=x, vertical=y.
        """
        matrices = mapdata_to_modelmatrix(raw_mapdata, MAP_ROWS, MAP_COLS)

        # RGB color for each mode
        mode_rgb = {
            "TG":  (1.00, 0.65, 0.00),  # orange
            "GG":  (0.00, 0.45, 1.00),  # blue
            "GSD": (0.00, 0.75, 0.00),  # green
            "TS":  (1.00, 0.00, 0.00),  # red
        }

        # Start with white background, shape (564, 529, 3)
        rgb = np.ones((MAP_COLS, MAP_ROWS, 3), dtype=np.float32)

        for mode_name in MODE_LIST:
            if mode_name not in matrices:
                continue
            grid = np.array(matrices[mode_name], dtype=np.float32)  # (564, 529)
            mask = grid > 0
            if not mask.any():
                continue
            r, g, b = mode_rgb.get(mode_name, (0.5, 0.5, 0.5))
            # Blend: take the mode color where this mode exists
            rgb[mask, 0] = np.minimum(rgb[mask, 0], r)
            rgb[mask, 1] = np.minimum(rgb[mask, 1], g)
            rgb[mask, 2] = np.minimum(rgb[mask, 2], b)

        # Transpose to (529, 564, 3) for imshow
        rgb = rgb.transpose(1, 0, 2)
        return rgb

    def _update_title(self):
        state = self.state
        match = state.current_match_rate()
        reached = "ARRIVED" if state.reached else "moving"
        title = (
            f"Mode: {state.mode} | Steps: {state.step_count} | "
            f"Dist: {state.remaining_dist:.1f} | "
            f"Match: {match:.2%} | {reached}"
        )
        self.ax.set_title(title, fontsize=11, fontfamily="monospace")

    def _draw_legend_box(self):
        text = (
            "Keys:\n"
            "  Arrow / WASD   move\n"
            "  Backspace      undo\n"
            "  R              reset\n"
            "  Enter          save & label"
        )
        self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )

    def show_label_prompt(self):
        """Show the label selection prompt after Enter is pressed."""
        self.ax.set_title(
            "SELECT LABEL:  [1] Expressway  [2] Nat/Prov  [3] HS-Rail  [4] Conv-Rail  [5] Other",
            fontsize=12, fontfamily="monospace", color="darkblue",
        )
        self.fig.canvas.draw_idle()

    def refresh(self):
        """Incremental update: path line + cursor position."""
        state = self.state
        xs = [p[0] for p in state.path_history]
        ys = [p[1] for p in state.path_history]
        self.path_line.set_data(xs, ys)
        self.cursor.set_offsets([[state.cur_x, state.cur_y]])
        self._update_title()
        self.fig.canvas.draw_idle()


# ========================== Controller ==========================

class LabelController:
    """Coordinates state, renderer, and keyboard input."""

    def __init__(self, state: LabelState, renderer: PathRenderer,
                 output_dir: str, batch_mode: bool, current_idx: int,
                 start_in_label_mode: bool = False):
        self.state = state
        self.renderer = renderer
        self.output_dir = output_dir
        self.batch_mode = batch_mode
        self.current_idx = current_idx
        self.saved = False
        self.selecting_label = False
        self.next_requested = False
        self.go_back_requested = False

        if start_in_label_mode:
            self.selecting_label = True
            self.renderer.show_label_prompt()
            print(f"  Select label: 1=Expressway 2=Nat/Prov 3=HS-Rail 4=Conv-Rail 5=Other")
            print(f"  (Backspace to re-edit path)")

    def on_key(self, event):
        if event.key is None:
            return

        key = event.key.lower()

        # --- label selection mode: 1-5 to pick, backspace to cancel ---
        if self.selecting_label:
            if key in LABEL_OPTIONS:
                label = LABEL_OPTIONS[key]
                self._finalize(label)
                self.selecting_label = False
                self.saved = True
                print(f"  [LABELED] #{self.current_idx} -> {label}")
                if self.batch_mode:
                    self.next_requested = True
                    plt.close(self.renderer.fig)
                else:
                    self.renderer.ax.set_title(
                        self.renderer.ax.get_title() + f" [{label}]",
                        fontsize=11, fontfamily="monospace",
                    )
                    self.renderer.fig.canvas.draw_idle()
            elif key == "backspace":
                # Cancel label selection, return to path editing
                self.selecting_label = False
                self.renderer._update_title()
                self.renderer.fig.canvas.draw_idle()
                print(f"  Label selection cancelled, back to path editing")
            return

        if key not in KEY_MAP:
            return

        action, arg = KEY_MAP[key]

        if action == "move":
            dx, dy = DX_DY[arg]
            if self.state.can_move(dx, dy):
                self.state.apply_move(arg)
                self.renderer.refresh()

        elif action == "undo":
            if len(self.state.path_history) <= 1:
                # No steps taken — go back to previous trajectory's label
                if self.current_idx > 0:
                    self.go_back_requested = True
                    plt.close(self.renderer.fig)
                    print(f"  Going back to re-label previous trajectory #{self.current_idx - 1}")
                else:
                    print(f"  Already at first trajectory, cannot go back")
            elif self.state.undo():
                self.renderer.refresh()

        elif action == "reset":
            self.state.reset()
            self.renderer.refresh()

        elif action == "save":
            self.selecting_label = True
            self.renderer.show_label_prompt()
            print(f"  Select label: 1=Expressway 2=Nat/Prov 3=HS-Rail 4=Conv-Rail 5=Other")
            print(f"  (Backspace to cancel)")

    def _finalize(self, label):
        """Write the complete record (path + label) to CSV and PNG."""
        state = self.state
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, "traj_labeled.csv")

        traj_list = [[float(p[0]), float(p[1])] for p in state.path_history]
        match_rate = calculate_match_rate(state.path_history, state.multi_mapdata)

        record = {
            "episode": self.current_idx,
            "order": state.order,
            "mode": state.mode,
            "label": label,
            "reward": 0.0,
            "success": 1 if state.reached else 0,
            "match": match_rate,
            "steps": state.step_count,
            "traj": json.dumps(traj_list, ensure_ascii=False),
        }

        df_new = pd.DataFrame([record])
        if os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8")
        else:
            df_new.to_csv(csv_path, index=False, encoding="utf-8")

        png_name = f"ep_{self.current_idx:04d}_order_{state.order}_{label}.png"
        png_path = os.path.join(self.output_dir, png_name)
        self.renderer.fig.savefig(png_path, bbox_inches="tight", dpi=150)
        print(f"  -> CSV: {csv_path}")
        print(f"  -> PNG: {png_path}")


# ========================== Main Loop ==========================

def run_single(state, raw_mapdata, output_dir, batch_mode, idx,
               start_in_label_mode=False):
    """Run labeling for one trajectory. Returns (next_idx, keep_going)."""
    renderer = PathRenderer(state, raw_mapdata)
    controller = LabelController(
        state, renderer, output_dir, batch_mode, idx,
        start_in_label_mode=start_in_label_mode,
    )

    renderer.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    def on_close(event):
        if not controller.saved:
            print(f"  [WARN] window closed, #{idx} not saved")

    renderer.fig.canvas.mpl_connect("close_event", on_close)

    if not start_in_label_mode:
        print(f"\n{'='*60}")
        print(f"#{idx}  order={state.order}  mode={state.mode}")
        print(f"Start: ({state.start_x}, {state.start_y})  ->  End: ({state.end_x}, {state.end_y})")
        print(f"Keys: Arrows/WASD=move  Backspace=undo  R=reset  Enter=save & label")
        print(f"{'='*60}")
    else:
        print(f"\n#{idx}  order={state.order}  mode={state.mode}  [RE-LABEL]")

    plt.show(block=True)

    if controller.next_requested:
        return idx + 1, True
    elif controller.go_back_requested:
        return max(0, idx - 1), True
    else:
        return idx, False


def main():
    parser = argparse.ArgumentParser(description="Interactive path labeling tool")
    parser.add_argument("--index", type=int, default=None,
                        help="label a single trajectory (0-based index)")
    parser.add_argument("--batch", action="store_true",
                        help="label all trajectories in sequence")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="output directory (default: label_output)")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH,
                        help="path to trajectory CSV")
    parser.add_argument("--mapdata", type=str, default=DEFAULT_MAPDATA_PATH,
                        help="path to map pickle file")
    args = parser.parse_args()

    print("Loading map data...")
    raw_mapdata = load_mapdata(args.mapdata)
    print("Loading trajectory data...")
    traj_df = load_traj_csv(args.csv)
    print(f"Total trajectories: {len(traj_df)}")

    output_dir = args.output

    if args.index is not None:
        if args.index < 0 or args.index >= len(traj_df):
            print(f"Error: index out of range [0, {len(traj_df)-1}]")
            sys.exit(1)
        row = traj_df.iloc[args.index]
        mode = str(row["mode"]).strip()
        multi = build_multi_mapdata(raw_mapdata, mode)
        state = LabelState(row, multi)
        run_single(state, raw_mapdata, output_dir, batch_mode=False, idx=args.index)

    elif args.batch:
        idx = 0
        while idx < len(traj_df):
            row = traj_df.iloc[idx]
            mode = str(row["mode"]).strip()
            multi = build_multi_mapdata(raw_mapdata, mode)
            state = LabelState(row, multi)
            next_idx, keep_going = run_single(
                state, raw_mapdata, output_dir, batch_mode=True, idx=idx,
            )
            if not keep_going:
                print(f"Labeling stopped, completed up to #{idx}")
                break
            # If go_back: next_idx < current idx, enter label mode on previous
            start_label = (next_idx < idx)
            idx = next_idx
            if start_label:
                row = traj_df.iloc[idx]
                mode = str(row["mode"]).strip()
                multi = build_multi_mapdata(raw_mapdata, mode)
                state = LabelState(row, multi)
                next_idx, keep_going = run_single(
                    state, raw_mapdata, output_dir, batch_mode=True, idx=idx,
                    start_in_label_mode=True,
                )
                if not keep_going:
                    break
                idx = next_idx
        if idx >= len(traj_df):
            print(f"\nAll {len(traj_df)} trajectories labeled!")

    else:
        # Prompt user for starting index
        while True:
            try:
                user_input = input(
                    f"Enter starting index [0-{len(traj_df)-1}], or press Enter for 0: "
                ).strip()
                if user_input == "":
                    start_idx = 0
                else:
                    start_idx = int(user_input)
                if 0 <= start_idx < len(traj_df):
                    break
                print(f"  Error: index out of range [0, {len(traj_df)-1}]")
            except ValueError:
                print(f"  Error: please enter a valid integer")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)

        idx = start_idx
        while idx < len(traj_df):
            row = traj_df.iloc[idx]
            mode = str(row["mode"]).strip()
            multi = build_multi_mapdata(raw_mapdata, mode)
            state = LabelState(row, multi)
            next_idx, keep_going = run_single(
                state, raw_mapdata, output_dir, batch_mode=True, idx=idx,
            )
            if not keep_going:
                print(f"Labeling stopped, completed up to #{idx}")
                break
            start_label = (next_idx < idx)
            idx = next_idx
            if start_label:
                row = traj_df.iloc[idx]
                mode = str(row["mode"]).strip()
                multi = build_multi_mapdata(raw_mapdata, mode)
                state = LabelState(row, multi)
                next_idx, keep_going = run_single(
                    state, raw_mapdata, output_dir, batch_mode=True, idx=idx,
                    start_in_label_mode=True,
                )
                if not keep_going:
                    break
                idx = next_idx
        if idx >= len(traj_df):
            print(f"\nAll {len(traj_df)} trajectories labeled!")


if __name__ == "__main__":
    main()
