#!/usr/bin/env python3
# Burst Scope for physical acoustic demonstations
# Uses a chosen loudspeaker as emitter and a chosen microphone as detector.
# Designed by Shahar Seifer, Weizmann Insitute of Science, 2025
# license: CC-BY-4.0  (https://creativecommons.org/licenses/by/4.0/legalcode)

# - Stage 1: estimates I/O latency using periodic bursts.
# - Stage 2: synchronized plot, with controls:
#     r : stops bursts, records 5s interval, searches for a short and hight peak and extracts baseline (window centered at peak)
#     1 : plays baseline once, shows microphone response and stores in memory
#     2 : plays reversed baseline once, records and shows microphone response
#     3 : plays deconvolution signal (Baseline/Response in frequency domain), records and shows response
#     q : quits program
#

import sys
import time
import queue
import threading

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: Requires 'sounddevice'. Install with: pip install sounddevice")
    sys.exit(1)

# -----------------------
# Configuration
# -----------------------
fs = 48000             # Sample rate (Hz)
freq = 3000.0          # Burst sine frequency (Hz)
cycles = 10            # Cycles per burst
period_ms = 1000.0     # Burst repetition period (ms)
output_gain = 0.2      # Output amplitude (0..1)
window_shape = 'none'  # 'none' or 'hann'

# Calibration (latency estimation)
latency_search_ms = 250.0
max_periods_back = 20
ema_alpha = 0.2
lock_std_ms = 2.0
lock_min_estimates = 1
min_corr_to_lock = 0.25

# Operation (plot and demos)
plot_window_ms = 40.0
plot_window_ms_TRM= 600
record_seconds = 5.0
refresh_interval_ms = int(period_ms)

deconv_lowpass_cutoff_hz = 500.0   # or whatever you want
deconv_lowpass_width_hz  = 100.0    # transition width for smooth roll-off

# Devices (None = defaults)
output_device = None
input_device = None  #use 1 for USB microphone when asked in the prompt

DEBUG = True

# -----------------------
# Derived params
# -----------------------
period_samples = int(round(fs * period_ms / 1000.0))
burst_samples = int(round(cycles * fs / freq))
plot_window_samples = int(round(fs * plot_window_ms / 1000.0))
search_half = int(round(fs * latency_search_ms / 1000.0))

# Ideal period (burst + zeros until period end)
n = np.arange(period_samples)
phase = 2 * np.pi * freq * n / fs
ideal_period = np.zeros(period_samples, dtype=np.float32)
ideal_period[:burst_samples] = np.sin(phase[:burst_samples]).astype(np.float32)
if window_shape.lower() == 'hann':
    ideal_period[:burst_samples] *= np.hanning(burst_samples).astype(np.float32)
ideal_period *= output_gain

# Compact ideal reference window (20 ms) for correlation
ref_len = int(round(fs * 0.02))
ref_win = np.zeros(ref_len, dtype=np.float32)
ref_take = min(ref_len, burst_samples)
ref_win[:ref_take] = ideal_period[:ref_take]
ref_win -= ref_win.mean()
ref_norm = np.linalg.norm(ref_win) + 1e-9

# Ring buffer for microphone
ring_seconds = max(record_seconds + 2.0, 12.0)
ring_len = int(ring_seconds * fs)
mic_ring = np.zeros(ring_len, dtype=np.float32)
ring_write_pos = 0
sample_counter = 0

# Queues/flags
info_q = queue.Queue(maxsize=1)
_go_to_operation = None  # None = unknown, True = go on, False = quit after calibration

latency_estimates = []
ema_latency = None

# Playback/Mode state
mode = 'burst'  # 'burst', 'idle', 'play_once'
play_buffer = None
play_pos = 0
play_start_sample = None

# Baseline / responses
baseline_win = None
response_win = None
stored_response_abs = None  # absolute position of response from pressing '1'
stored_response_win = None  # waveform of response captured after key '1'

# Global lock for shared state
state_lock = threading.Lock()


# -----------------------
# Audio callback
# -----------------------

def audio_callback(indata, outdata, frames, time_info, status):
    global ring_write_pos, mic_ring, sample_counter, play_pos, play_start_sample, mode
    if status:
        print(status, file=sys.stderr)

    # Copy state under lock to local variables (short lock)
    with state_lock:
        local_mode = mode
        local_play_buffer = play_buffer
        local_play_pos = play_pos
        local_sample_counter = sample_counter
        local_ring_write_pos = ring_write_pos

    # Output selection
    if local_mode == 'burst':
        start_pos = local_sample_counter
        idx = (np.arange(frames, dtype=np.int64) + start_pos) % period_samples
        out = ideal_period[idx]
    elif local_mode == 'play_once' and local_play_buffer is not None:
        remaining = len(local_play_buffer) - local_play_pos
        out = np.zeros(frames, dtype=np.float32)
        take = min(frames, remaining)
        if take > 0:
            out[:take] = local_play_buffer[local_play_pos:local_play_pos + take]
            local_play_pos += take
            if local_play_pos >= len(local_play_buffer):
                with state_lock:
                    mode = 'idle'
        else:
            with state_lock:
                mode = 'idle'
    else:
        out = np.zeros(frames, dtype=np.float32)

    # Write to output (mono or duplicate)
    if outdata.shape[1] == 1:
        outdata[:, 0] = out
    else:
        outdata[:] = out[:, None]

    # Record microphone (first channel)
    mic_block = indata[:, 0].astype(np.float32)

    # Update ring buffer and counters under lock
    with state_lock:
        end_pos = ring_write_pos + frames
        if end_pos <= ring_len:
            mic_ring[ring_write_pos:end_pos] = mic_block
        else:
            first = ring_len - ring_write_pos
            mic_ring[ring_write_pos:] = mic_block[:first]
            mic_ring[:end_pos - ring_len] = mic_block[first:]
        ring_write_pos = end_pos % ring_len
        sample_counter += frames
        if local_mode == 'play_once' and play_buffer is not None:
            play_pos = local_play_pos
        now_samples = sample_counter

    # Notify UI
    if not info_q.full():
        info_q.put(now_samples)


# -----------------------
# Helpers
# -----------------------

def ring_read_abs(abs_start, length):
    """Read from mic_ring in absolute sample coordinates."""
    out = np.empty(length, dtype=np.float32)
    with state_lock:
        local_ring_write_pos = ring_write_pos
        local_sample_counter = sample_counter
        local_mic_ring = mic_ring.copy()
    for i in range(length):
        k = abs_start + i
        ring_idx = (local_ring_write_pos - (local_sample_counter - k)) % ring_len
        out[i] = local_mic_ring[ring_idx]
    return out


def normxcorr(segment, ref):
    seg = segment.astype(np.float32)
    seg -= seg.mean()
    denom = (np.linalg.norm(seg) + 1e-9) * ref_norm
    return float(np.dot(seg, ref) / denom)


def estimate_latency_once(now_counter):
    base = (now_counter // period_samples) * period_samples
    best_corr = -1.0
    best_latency = 0
    for k in range(max_periods_back):
        burst_start = base - k * period_samples
        center = burst_start + (int(ema_latency) if ema_latency is not None else search_half)
        start = center - search_half
        length = 2 * search_half + ref_len
        if length <= 0:
            continue
        mic_win = ring_read_abs(start, length)
        for lag in range(0, 2 * search_half + 1):
            seg = mic_win[lag:lag + ref_len]
            c = normxcorr(seg, ref_win)
            if c > best_corr:
                best_corr = c
                best_latency = (start + lag) - burst_start
    return best_latency, best_corr


def push_latency(lat_samples):
    global ema_latency
    if ema_latency is None:
        ema_latency = float(lat_samples)
    else:
        ema_latency = (1 - ema_alpha) * ema_latency + ema_alpha * float(lat_samples)
    latency_estimates.append(float(lat_samples))
    if len(latency_estimates) > 120:
        del latency_estimates[:-120]


def latency_ms_stats():
    if not latency_estimates:
        return None, None
    ms_vals = np.array(latency_estimates) * 1000.0 / fs
    return float(np.mean(ms_vals)), float(np.std(ms_vals))


# -----------------------
# Calibration (Stage 1)
# -----------------------

def run_calibration():
    global _go_to_operation
    _go_to_operation = None

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        fig.canvas.manager.set_window_title('Calibration – Latency estimation')
    except Exception:
        pass
    ax.axis('off')
    txt1 = ax.text(0.02, 0.85, '', transform=ax.transAxes, va='top', fontsize=11)
    txt2 = ax.text(0.02, 0.60, '', transform=ax.transAxes, va='top', fontsize=10)
    ax.text(0.02, 0.35,
            'Press S to switch to Operation. Press Q to quit.',
            transform=ax.transAxes, va='top', fontsize=10)

    last_print = time.time()

    def update():
        nonlocal last_print
        try:
            now = info_q.get_nowait()
        except queue.Empty:
            with state_lock:
                now = sample_counter
        lat_samples, corr = estimate_latency_once(now)
        push_latency(lat_samples)
        mean_ms, std_ms = latency_ms_stats()
        line1 = f"corr={corr:.3f}  latest_latency={lat_samples*1000.0/fs:.2f} ms"
        if mean_ms is not None:
            line2 = (f"EMA={ema_latency*1000.0/fs:.2f} ms  "f"mean±std (last {len(latency_estimates)}): {mean_ms:.2f} ± {std_ms:.2f} ms")
        else:
            line2 = "Collecting estimates..."
        txt1.set_text(line1)
        txt2.set_text(line2)
        fig.canvas.draw_idle()
        if DEBUG and time.time() - last_print > 1.0:
            print(line1 + " | " + line2)
            last_print = time.time()
        # Auto-lock with guard
        if (mean_ms is not None
                and len(latency_estimates) >= lock_min_estimates
                and std_ms <= lock_std_ms
                and corr >= min_corr_to_lock):
            _go_to_operation = True
            plt.close(fig)

    timer = fig.canvas.new_timer(interval=refresh_interval_ms)
    timer.add_callback(update)
    timer.start()

    def on_key(event):
        global _go_to_operation
        if not event.key:
            return
        k = event.key.lower()
        if k == 's':
            _go_to_operation = True
            plt.close(fig)
        elif k == 'q':
            _go_to_operation = False
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# -----------------------
# Operation (Stage 2) – two vertical plots
# -----------------------

def run_operation():
    global mode, baseline_win, response_win, play_buffer, play_pos, play_start_sample
    global stored_response_abs, stored_response_win

    # High‑level operation state (independent from audio 'mode')
    # op_state: 'burst', 'recording_baseline', 'playing_baseline',
    #           'playing_reverse', 'playing_deconv', 'frozen', 'idle'
    op_state = 'burst'
    pending_cmd = None          # 'record_baseline', 'play_baseline', 'play_reverse', 'play_deconv', 'idle'
    last_play_kind = None       # 'baseline', 'reverse', 'deconv'

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax_tx, ax_rx) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    try:
        fig.canvas.manager.set_window_title('Operation – Tx/Rx view & demo')
    except Exception:
        pass

    t = np.arange(plot_window_samples) / fs * 1000.0

    # --- Transmit plot (fixed scale) ---
    tx_line, = ax_tx.plot(t, np.zeros_like(t), color='tab:blue')
    ax_tx.set_ylabel('Tx signal')
    ax_tx.set_title('Transmit signal')
    ax_tx.set_ylim(-1.1 * output_gain, 1.1 * output_gain)

    # --- Response plot (auto-scale) ---
    rx_line, = ax_rx.plot(t, np.zeros_like(t), color='tab:orange')
    ax_rx.set_ylabel('Rx signal')
    ax_rx.set_xlabel('Time (ms)')
    ax_rx.set_title('Microphone response')

    text = ax_rx.text(0.02, 0.95, '', transform=ax_rx.transAxes, va='top', fontsize=10)

    # --- Local UI state ---
    play_lock = threading.Lock()
    view_start_abs = None       # where to read response from (absolute sample)
    tx_view = np.zeros(plot_window_samples, dtype=np.float32)

    frozen_tx = np.zeros(plot_window_samples, dtype=np.float32)
    frozen_rx = np.zeros(plot_window_samples, dtype=np.float32)

    # Baseline recording state (sample counting)
    baseline_recording = False
    baseline_start_abs = None
    baseline_target_samples = int(record_seconds * fs)

    # For response capture after playback
    response_capture_abs = None  # absolute sample index of start of response window

    def plot_status(msg):
        text.set_text(msg)

    def align_mic_for_burst(now_samples):
        """Show mic around latest burst using EMA latency."""
        with state_lock:
            lat = int(round(ema_latency)) if ema_latency is not None else 0
        last_burst = (now_samples // period_samples) * period_samples
        start = last_burst + lat
        return ring_read_abs(start, plot_window_samples)

    def extract_and_freeze_baseline(start_abs, end_abs):
        """Cut 10 s recording to a baseline window centered at peak, freeze it."""
        nonlocal op_state, view_start_abs, tx_view, frozen_tx, frozen_rx
        global baseline_win, response_win, stored_response_abs

        rec = ring_read_abs(start_abs, end_abs - start_abs)
        i_max = int(np.argmax(np.abs(rec)))
        start_win = start_abs + i_max - plot_window_samples // 2
        start_win = max(start_abs, start_win)
        end_win = start_win + plot_window_samples
        if end_win > end_abs:
            end_win = end_abs
            start_win = end_win - plot_window_samples

        new_baseline = ring_read_abs(start_win, plot_window_samples)

        # Store global baseline, clear old response info
        baseline_win = new_baseline.copy()
        stored_response_abs = None
        response_win = None

        view_start_abs = start_win
        tx_view = new_baseline.copy()

        frozen_tx = new_baseline.copy()
        frozen_rx = ring_read_abs(start_win, plot_window_samples)
        op_state = 'frozen'

        plot_status('Baseline ready and frozen. Keys: 1=play baseline, 2=reversed, 3=deconv (uses response from 1), q=quit')

    # --- Update loop ---
    def update():
        try:
            _update_core()
        except Exception as e:
            print("EXCEPTION IN update():", repr(e))
            import traceback
            traceback.print_exc()
            plt.close('all')
    def _update_core():
        global mode, play_buffer, play_pos, play_start_sample, stored_response_win
        nonlocal op_state, pending_cmd, last_play_kind
        nonlocal baseline_recording, baseline_start_abs
        nonlocal view_start_abs, tx_view
        nonlocal frozen_tx, frozen_rx, response_capture_abs

        # If plot_window_samples changed, rebuild arrays and x-axis
        if len(tx_view) != plot_window_samples:
            t_new = np.arange(plot_window_samples) / fs * 1000.0
            tx_line.set_xdata(t_new)
            rx_line.set_xdata(t_new)
            tx_view = np.zeros(plot_window_samples, dtype=np.float32)
            frozen_tx = np.zeros(plot_window_samples, dtype=np.float32)
            frozen_rx = np.zeros(plot_window_samples, dtype=np.float32)
        # Get latest sample counter from audio thread
        try:
            now = info_q.get_nowait()
        except queue.Empty:
            with state_lock:
                now = sample_counter

        # Snapshot global audio state
        with state_lock:
            local_mode = mode
            local_sample_counter = sample_counter

        # -------------------------------------------------
        # 1) Handle pending command (event‑driven)
        # -------------------------------------------------
        if pending_cmd is not None:
            cmd = pending_cmd
            pending_cmd = None

            if cmd == 'idle':
                with state_lock:
                    mode = 'idle'
                op_state = 'idle'
                frozen_tx[:] = 0
                frozen_rx[:] = 0
                plot_status('Idle. Press r to record baseline.')
                # fall through to drawing

            elif cmd == 'record_baseline':
                # Start 10 s baseline recording
                with state_lock:
                    mode = 'idle'  # stop bursts during baseline record
                    baseline_start_abs = sample_counter
                baseline_recording = True
                op_state = 'recording_baseline'
                plot_status(f'Recording baseline for {record_seconds:.1f} s...')

            elif cmd == 'play_baseline':
                if baseline_win is None:
                    plot_status('No baseline yet. Press r first.')
                else:
                    current_baseline = baseline_win.copy()
                    with play_lock, state_lock:
                        play_buffer = current_baseline
                        play_pos = 0
                        play_start_sample = sample_counter
                        mode = 'play_once'
                        lat = int(round(ema_latency)) if ema_latency is not None else 0

                    response_capture_abs = play_start_sample + lat
                    view_start_abs = response_capture_abs
                    tx_view = current_baseline.copy()
                    op_state = 'playing_baseline'
                    last_play_kind = 'baseline'
                    plot_status('Baseline playing... response will freeze automatically.')

            elif cmd == 'play_reverse':
                if baseline_win is None:
                    plot_status('No baseline yet. Press r first.')
                else:
                    current_baseline = baseline_win.copy()
                    rev = current_baseline[::-1].copy()
                    with play_lock, state_lock:
                        play_buffer = rev
                        play_pos = 0
                        play_start_sample = sample_counter
                        mode = 'play_once'
                        lat = int(round(ema_latency)) if ema_latency is not None else 0

                    response_capture_abs = play_start_sample + lat
                    view_start_abs = response_capture_abs
                    tx_view = rev.copy()
                    op_state = 'playing_reverse'
                    last_play_kind = 'reverse'
                    plot_status('Reversed baseline playing... response will freeze automatically.')

            elif cmd == 'play_deconv':
                if baseline_win is None:
                    plot_status('No baseline yet. Press r first.')
                elif stored_response_win is None:
                    plot_status('No stored response from key 1. Press 1 first.')
                else:
                    current_baseline = baseline_win.copy()
                    response_once = stored_response_win.copy()

                    flag_need_centering=False
                    # Center response window on its peak
                    if flag_need_centering:
                        i_max = int(np.argmax(np.abs(response_once)))
                        start = i_max - plot_window_samples // 2
                        start = max(0, start)
                        end = start + plot_window_samples
                        if end > len(response_once):
                            end = len(response_once)
                            start = end - plot_window_samples
                        response_centered = response_once[start:end]
                    else:
                        response_centered = response_once
                    # --- Remove DC offset from both signals ---
                    current_baseline = current_baseline - np.mean(current_baseline)
                    response_centered = response_centered - np.mean(response_centered)

                    N = plot_window_samples
                    B = np.fft.rfft(current_baseline, n=N)
                    R = np.fft.rfft(response_centered, n=N)
                    eps = 1e-8 * np.max(np.abs(R))
                    H = B / (R + eps)

                    flag_need_low_pass_filtering=False
                    # --- Low-pass filtering of H in frequency domain ---
                    if flag_need_low_pass_filtering:
                        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
                        fc = deconv_lowpass_cutoff_hz
                        fw = deconv_lowpass_width_hz
                        # Smooth mask: 1 below (fc - fw), 0 above (fc + fw), raised-cosine in between
                        mask = np.ones_like(freqs, dtype=np.float32)
                        if fw > 0:
                            # transition band logical indices
                            lo = fc - fw
                            hi = fc + fw
                            # below lo = 1, above hi = 0
                            mask[freqs >= hi] = 0.0
                            # smooth transition in [lo, hi]
                            trans = (freqs >= lo) & (freqs < hi)
                            x = (freqs[trans] - lo) / (hi - lo)  # 0..1
                            mask[trans] = 0.5 * (1.0 + np.cos(np.pi * x))  # raised cosine from 1 to 0
                        else:
                            mask[freqs > fc] = 0.0
                        H_filtered = H * mask
                    else:
                        H_filtered = H

                    h = np.fft.irfft(H_filtered, n=N).astype(np.float32)
                    # Shift so the peak appears in the center
                    h_centered = np.fft.fftshift(h)

                    m = np.max(np.abs(h_centered)) + 1e-9
                    tx = (output_gain / m) * h_centered

                    with play_lock, state_lock:
                        play_buffer = tx
                        play_pos = 0
                        play_start_sample = sample_counter
                        mode = 'play_once'
                        lat = int(round(ema_latency)) if ema_latency is not None else 0

                    response_capture_abs = play_start_sample + lat
                    view_start_abs = response_capture_abs
                    tx_view = tx.copy()
                    op_state = 'playing_deconv'
                    last_play_kind = 'deconv'

        # -------------------------------------------------
        # 2) Baseline recording completion (sample counting)
        # -------------------------------------------------
        if op_state == 'recording_baseline' and baseline_recording and baseline_start_abs is not None:
            if local_sample_counter - baseline_start_abs >= baseline_target_samples:
                start_abs = baseline_start_abs
                end_abs = baseline_start_abs + baseline_target_samples
                baseline_recording = False
                extract_and_freeze_baseline(start_abs, end_abs)
                # op_state becomes 'frozen' inside helper

        # -------------------------------------------------
        # 3) Playback completion → capture response & freeze
        # -------------------------------------------------
        if op_state in ('playing_baseline', 'playing_reverse', 'playing_deconv'):
            if (local_mode == 'idle'
                    and response_capture_abs is not None
                    and local_sample_counter >= response_capture_abs + plot_window_samples):

                mic = ring_read_abs(response_capture_abs, plot_window_samples)
                frozen_tx = tx_view.copy()
                frozen_rx = mic.copy()
                op_state = 'frozen'

                if last_play_kind == 'baseline':
                    stored_response_win = mic.copy()

                plot_status('Playback finished. Response captured and frozen.')
        # -------------------------------------------------
        # 4) Drawing, depending on op_state
        # -------------------------------------------------
        if op_state == 'burst':
            # Show synchronized burst view
            mic = align_mic_for_burst(now)
            ideal = np.tile(ideal_period, 1 + plot_window_samples // period_samples)[:plot_window_samples]

        elif op_state == 'recording_baseline':
            # Optional: show the last part of the recording; or just silence
            if baseline_start_abs is not None:
                # show most recent window as a rough preview
                start = max(baseline_start_abs, local_sample_counter - plot_window_samples)
                mic = ring_read_abs(start, plot_window_samples)
            else:
                mic = np.zeros(plot_window_samples, dtype=np.float32)
            ideal = np.zeros(plot_window_samples, dtype=np.float32)

        elif op_state in ('playing_baseline', 'playing_reverse', 'playing_deconv'):
            # Show live response while playing
            if response_capture_abs is not None:
                mic = ring_read_abs(response_capture_abs, plot_window_samples)
            else:
                mic = np.zeros(plot_window_samples, dtype=np.float32)
            ideal = tx_view

        elif op_state == 'frozen':
            mic = frozen_rx
            ideal = frozen_tx

        elif op_state == 'idle':
            mic = np.zeros(plot_window_samples, dtype=np.float32)
            ideal = np.zeros(plot_window_samples, dtype=np.float32)

        else:
            # Fallback – should not happen
            mic = np.zeros(plot_window_samples, dtype=np.float32)
            ideal = np.zeros(plot_window_samples, dtype=np.float32)

        tx_line.set_ydata(ideal)
        rx_line.set_ydata(mic)

        if op_state == 'frozen':
            ax_rx.relim()
            ax_rx.autoscale_view()

        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=refresh_interval_ms)
    timer.add_callback(update)
    timer.start()

    # --- Key handler: only sets commands / simple state ---
    def on_key(event):
        nonlocal pending_cmd, op_state
        global mode, play_buffer, play_pos, play_start_sample, stored_response_abs, stored_response_win
        global plot_window_ms, plot_window_samples
        if not event.key:
            return
        k = event.key.lower()

        if k == 'q':
            plt.close(fig)
            return

        if k == '0':
            pending_cmd = 'idle'
            return

        if k == 'r':
            plot_window_ms = plot_window_ms_TRM
            plot_window_samples = int(round(fs * plot_window_ms / 1000.0))
            pending_cmd = 'record_baseline'
            return

        if k == '1':
            pending_cmd = 'play_baseline'
            return

        if k == '2':
            pending_cmd = 'play_reverse'
            return

        if k == '3':
            pending_cmd = 'play_deconv'
            return

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Start in burst mode visually; audio 'mode' is already 'burst' globally
    op_state = 'burst'
    plot_status('Burst mode. Press r to record baseline, 1/2/3 to play, q to quit.')

    plt.show()


def ask_device(prompt):
    s = input(prompt).strip()
    return None if s == "" else int(s)
def choose_devices(devices):
    root = tk.Tk()
    root.title("Select Devices")

    tk.Label(root, text="Output Device:").pack(pady=5)
    combo_out = ttk.Combobox(root, values=devices, width=80)
    combo_out.pack(pady=5)
    combo_out.current(0)

    tk.Label(root, text="Input Device:").pack(pady=5)
    combo_in = ttk.Combobox(root, values=devices, width=80)
    combo_in.pack(pady=5)
    combo_in.current(0)

    result = {"out": None, "in": None}

    def ok():
        out_sel = combo_out.get().split(":")[0]
        in_sel  = combo_in.get().split(":")[0]
        result["out"] = None if out_sel == "None" else int(out_sel)
        result["in"]  = None if in_sel  == "None" else int(in_sel)
        root.destroy()

    tk.Button(root, text="OK", command=ok).pack(pady=10)
    root.mainloop()

    return result["out"], result["in"]


# -----------------------
# Main
# -----------------------

def main():
    print(f"fs={fs} Hz, freq={freq} Hz, cycles={cycles}, period={period_ms} ms, window={plot_window_ms} ms")
    print(f"In the dialog box choose audio devices, use None for default.")
    # Build device list
    devs = sd.query_devices()
    device_list = ["None"] + [f"{i}: {dev['name']}" for i, dev in enumerate(devs)]
    output_device, input_device = choose_devices(device_list)

    print("Output:", output_device)
    print("Input:", input_device)

    if input_device is not None or output_device is not None:
        sd.default.device = (input_device, output_device)
    sd.default.samplerate = fs

    stream = sd.Stream(
        samplerate=fs,
        blocksize=0,
        dtype='float32',
        channels=(1, 1),
        callback=audio_callback,
        dither_off=True,
        latency=None,
        device=(input_device, output_device)
    )

    with stream:
        print("Stream active. Calibration estimates latency...")
        time.sleep(0.3)
        run_calibration()

        if _go_to_operation is False:
            print("User chose to quit after calibration.")
            return

        print("Switching to Operation. Press q in the window to exit.")
        run_operation()

    print("Exiting program.")


if __name__ == '__main__':
    main()