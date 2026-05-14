# =============================================================================
# Combined Experiment: Social Simon Task + Interpersonal Synchrony Tapping Task
# =============================================================================
#
# STRUCTURE (fixed order):
#   1. Participant 1 — Individual Go/No-Go Simon Task
#   2. Participant 2 — Individual Go/No-Go Simon Task
#   3. Both participants — Synchrony/Asynchrony Tapping Task
#   4. Both participants — Joint Go/No-Go Simon Task
#
# KEYS (consistent throughout):
#   Participant 1 → 'z'    (responds to SQUARE in Simon tasks)
#   Participant 2 → 'm'    (responds to DIAMOND in Simon tasks)
#
# OUTPUT:
#   data/dyad{ID}_simon_{timestamp}.csv        — all Simon data (individual + joint)
#   data/dyad{ID}_synchrony_{timestamp}.csv    — synchrony/asynchrony tapping data
#     (one row per tap, per participant)
#   data/dyad{ID}_participants_{timestamp}.csv — consent + demographics (age, gender)
#
# REQUIREMENTS:
#   pip install psychopy
#
# USAGE:
#   python experiment.py
# =============================================================================

from psychopy import core, visual, sound, event, gui
import numpy as np
import os
import csv
from datetime import datetime

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

# Keys — fixed across all tasks
KEY_P1   = 'z'
KEY_P2   = 'm'
QUIT_KEY = 'escape'

# Simon task shapes assigned to each participant
SHAPE_P1 = 'square'
SHAPE_P2 = 'diamond'

OUTPUT_DIR = 'data'

# -----------------------------------------------------------------------------
# Simon task timing (seconds)
# -----------------------------------------------------------------------------
FIXATION_PRE          = 0.250
STIM_DUR              = 0.150
RESPONSE_WIN          = 1.800
FEEDBACK_DUR          = 0.300
ITI_DUR               = 1.750 # if variable: np.random.uniform(1.5, 2.0)
PAUSE_BETWEEN_BLOCKS  = 1.000 # seconds pause after pressing SPACE before next block starts

# Simon block sizes
PRACTICE_TRIALS       = 10
EXP_TRIALS_PER_BLOCK  = 30
EXP_BLOCKS            = 2

# Simon stimulus geometry (height units)
STIM_LEFT_POS  = (-0.65, 0.0)
STIM_RIGHT_POS = ( 0.65, 0.0)
STIM_SIZE      = 0.12

# -----------------------------------------------------------------------------
# Synchrony/Asynchrony tapping task
# -----------------------------------------------------------------------------
BPM_SYNC          = 120
BPM_P1_ASYNC      = 120
BPM_P2_ASYNC      = 97        # no simple ratio with 120; lowest common multiple ~148 s

TASK_DURATION_S   = 90
WARMUP_DURATION_S = 10

CLICK_DURATION_S  = 0.04
CLICK_FREQ_P1_HZ  = 1200      # high pitch → left ear (Participant 1)
CLICK_FREQ_P2_HZ  = 400       # low pitch  → right ear (Participant 2)
CLICK_VOLUME      = 0.6


# =============================================================================
# SHARED WINDOW  (created once, reused across all tasks)
# =============================================================================

def make_window():
    return visual.Window(
        size    = [1024, 768],
        fullscr = True,
        color   = 'black',
        units   = 'height',
        winType = 'pyglet',
    )


# =============================================================================
# QUIT HELPER
# =============================================================================

def quit_experiment(win, file_handles):
    for fh in file_handles.values():
        try:
            fh.close()
        except Exception:
            pass
    win.close()
    core.quit()


# =============================================================================
# SIMON TASK HELPERS
# =============================================================================

def draw_fixation(win):
    for start, end in [( (-0.02,0),(0.02,0) ), ( (0,-0.02),(0,0.02) )]:
        visual.Line(win, start=start, end=end,
                    lineColor='white', lineWidth=3, units='height').draw()


def draw_square(win, pos):
    visual.Rect(win, width=STIM_SIZE, height=STIM_SIZE,
                pos=pos, lineColor='white', fillColor=None,
                lineWidth=3, units='height').draw()


def draw_diamond(win, pos):
    visual.Rect(win, width=STIM_SIZE, height=STIM_SIZE,
                pos=pos, ori=45, lineColor='white', fillColor=None,
                lineWidth=3, units='height').draw()


def build_simon_trial_list(n_trials):
    """Balanced, randomised list of {shape, side} dicts."""
    shapes = ['square', 'diamond'] * (n_trials // 2)
    sides  = ['left',   'right'  ] * (n_trials // 2)
    np.random.shuffle(shapes)
    np.random.shuffle(sides)
    return [{'shape': s, 'side': d} for s, d in zip(shapes, sides)]


# =============================================================================
# SYNCHRONY TASK HELPERS
# =============================================================================

def make_click(freq, duration=CLICK_DURATION_S, volume=CLICK_VOLUME, pan='left'):
    """Synthesise a stereo click panned to one ear."""
    sample_rate = 44100
    t     = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone  = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
    stereo = np.zeros((len(tone), 2), dtype=np.float32)
    if pan == 'left':
        stereo[:, 0] = tone
    else:
        stereo[:, 1] = tone
    return sound.Sound(stereo, sampleRate=sample_rate, stereo=True)


def nearest_beat(tap_time, beat_times):
    """Return (closest beat time, signed asynchrony in ms)."""
    if len(beat_times) == 0:
        return np.nan, np.nan
    diffs = np.array(beat_times) - tap_time
    idx   = np.argmin(np.abs(diffs))
    return beat_times[idx], diffs[idx] * 1000.0


# =============================================================================
# TASK 1 & 2 — INDIVIDUAL SIMON
# =============================================================================

def run_individual_simon(win, participant, dyad_id, simon_writer, file_handles):
    """
    Run the individual go/no-go Simon task for one participant.
    participant : 1 or 2
    """
    key        = KEY_P1 if participant == 1 else KEY_P2
    go_shape   = SHAPE_P1 if participant == 1 else SHAPE_P2
    nogo_shape = SHAPE_P2 if participant == 1 else SHAPE_P1

    clock    = core.Clock()
    msg      = visual.TextStim(win, text='', color='white',
                               height=0.05, wrapWidth=0.9)

    def show_text(text, wait_key=True):
        msg.setText(text)
        msg.draw()
        win.flip()
        if wait_key:
            keys = event.waitKeys(keyList=['space', QUIT_KEY])
            if QUIT_KEY in keys:
                quit_experiment(win, file_handles)

    def run_block(block_num, trials, is_practice=False):
        for trial_num, trial in enumerate(trials, start=1):
            shape = trial['shape']
            side  = trial['side']
            pos   = STIM_LEFT_POS if side == 'left' else STIM_RIGHT_POS
            is_go = (shape == go_shape)

            # Flush anything pressed during the previous trial's feedback/ITI
            event.clearEvents()
            
            # Fixation
            draw_fixation(win)
            win.flip()
            core.wait(FIXATION_PRE)

            # Stimulus onset
            clock.reset()
            response_key = None
            rt_ms        = None
            responded    = False

            draw_fixation(win)
            if shape == 'square':
                draw_square(win, pos)
            else:
                draw_diamond(win, pos)
            win.flip()
            stim_onset = clock.getTime()

            while clock.getTime() - stim_onset < RESPONSE_WIN:
                t = clock.getTime() - stim_onset
                if t >= STIM_DUR and not responded:
                    draw_fixation(win)
                    win.flip()

                keys = event.getKeys(
                    keyList=[KEY_P1, KEY_P2, QUIT_KEY],
                    timeStamped=clock,
                )
                for k, kt in keys:
                    if k == QUIT_KEY:
                        quit_experiment(win, file_handles)
                    if not responded:
                        response_key = k
                        rt_ms        = (kt - stim_onset) * 1000
                        responded    = True
                        break
                if responded:
                    break

            # Flush any keys that arrived after the first response
            # (e.g. co-participant pressing their key) so they don't
            # bleed into the next trial.
            event.clearEvents()

            # Outcome
            if not responded:
                correct  = not is_go          # correct if no-go, error if go
                feedback = 'fixation' if correct else 'too_slow'
            else:
                correct  = (response_key == key) and is_go
                feedback = 'fixation' if correct else 'error'

            # Feedback
            if feedback == 'fixation':
                draw_fixation(win)
            else:
                msg.setText('Too slow' if feedback == 'too_slow' else 'Error')
                msg.draw()
            win.flip()
            core.wait(FEEDBACK_DUR)

            # ITI
            win.flip()
            core.wait(ITI_DUR)

            # Log (skip practice)
            if not is_practice:
                simon_writer.writerow([
                    dyad_id, trial_num, block_num, 'individual', participant,
                    shape, side, response_key or '',
                    int(responded), int(correct),
                    f'{rt_ms:.2f}' if rt_ms is not None else '',
                    feedback,
                ])

    # Instructions
    show_text(
        f"INDIVIDUAL GO/NO-GO TASK - Participant {participant}\n\n"
        f"Press  [ {key.upper()} ]  when you see a  {go_shape.upper()}\n"
        f"Do NOT press anything when you see a  {nogo_shape.upper()}\n\n"
        f"Respond as quickly and accurately as possible.\n\n"
        f"Press SPACE to begin the practice block."
    )
    win.flip()          # blank screen
    core.wait(PAUSE_BETWEEN_BLOCKS)  # short pause before first trial

    # Practice
    run_block(0, build_simon_trial_list(PRACTICE_TRIALS), is_practice=True)

    show_text(
        f"Practice complete!\n\n"
        f"You have {EXP_BLOCKS} blocks of {EXP_TRIALS_PER_BLOCK} trials each.\n\n"
        f"Press SPACE to start Block 1."
    )
    win.flip()          # blank screen
    core.wait(PAUSE_BETWEEN_BLOCKS)  # short pause before first trial

    # Experimental blocks
    for b in range(1, EXP_BLOCKS + 1):
        run_block(b, build_simon_trial_list(EXP_TRIALS_PER_BLOCK))
        if b < EXP_BLOCKS:
            show_text(
                f"Block {b} complete - take a short break.\n\n"
                f"Press SPACE when ready for Block {b + 1}."
            )
            win.flip()          # blank screen
            core.wait(PAUSE_BETWEEN_BLOCKS)  # short pause before first trial

    show_text(
        f"Individual task complete for Participant {participant}.\n\n"
        f"Press SPACE to continue."
    )


# =============================================================================
# TASK 3 — SYNCHRONY / ASYNCHRONY TAPPING
# =============================================================================

def run_synchrony_task(win, condition, dyad_id, sync_writer, file_handles):
    """
    condition : 'synchrony' or 'asynchrony'
    """
    if condition == 'synchrony':
        bpm_p1, bpm_p2 = BPM_SYNC, BPM_SYNC
    else:
        bpm_p1, bpm_p2 = BPM_P1_ASYNC, BPM_P2_ASYNC

    ibi_p1   = 60.0 / bpm_p1
    ibi_p2   = 60.0 / bpm_p2
    beats_p1 = np.arange(0, TASK_DURATION_S + ibi_p1, ibi_p1)
    beats_p2 = np.arange(0, TASK_DURATION_S + ibi_p2, ibi_p2)

    click_p1 = make_click(freq=CLICK_FREQ_P1_HZ, pan='left')
    click_p2 = make_click(freq=CLICK_FREQ_P2_HZ, pan='right')

    clock = core.Clock()
    msg   = visual.TextStim(win, text='', color='white',
                            height=0.04, wrapWidth=0.9)

    def show_message(text, wait_key=True):
        msg.setText(text)
        msg.draw()
        win.flip()
        if wait_key:
            keys = event.waitKeys(keyList=['space', QUIT_KEY])
            if QUIT_KEY in keys:
                quit_experiment(win, file_handles)

    cond_label = 'TOGETHER' if condition == 'synchrony' else 'SEPARATE RHYTHMS'

    if condition == 'synchrony':
        p1_instr = f"Participant 1  [ {KEY_P1.upper()} ]  → tap in time with the metronome"
        p2_instr = f"Participant 2  [ {KEY_P2.upper()} ]  → tap in time with the metronome"
    else:
        p1_instr = f"Participant 1  [ {KEY_P1.upper()} ]  → tap in time with the HIGH-PITCH click (left ear)"
        p2_instr = f"Participant 2  [ {KEY_P2.upper()} ]  → tap in time with the LOW-PITCH click (right ear)"

    show_message(
        f"SYNCHRONY TAPPING TASK - {cond_label}\n\n"
        f"{p1_instr}\n"
        f"{p2_instr}\n\n"
        f"A {WARMUP_DURATION_S}s warm-up will play first, then {TASK_DURATION_S}s of recording.\n\n"
        f"Press SPACE to begin."
    )
    # Warm-up
    msg.setText("Warm-up - tap along!\n\nRecording starts soon…")
    msg.draw()
    win.flip()

    clock.reset()
    next_p1, next_p2 = 0.0, 0.0
    while clock.getTime() < WARMUP_DURATION_S:
        t = clock.getTime()
        if t >= next_p1:
            click_p1.play()
            next_p1 += ibi_p1
        if condition == 'asynchrony' and t >= next_p2:
            click_p2.play()
            next_p2 += ibi_p2
        event.getKeys()

    # Recording
    tap_data = {1: [], 2: []}
    msg.setText(
        f"Recording...\n\n"
        f"[ {KEY_P1.upper()} ]  Participant 1          Participant 2  [ {KEY_P2.upper()} ]"
    )

    clock.reset()
    next_p1, next_p2 = 0.0, 0.0
    while True:
        t = clock.getTime()
        if t >= TASK_DURATION_S:
            break

        if t >= next_p1:
            click_p1.play()
            next_p1 += ibi_p1
        if condition == 'asynchrony' and t >= next_p2:
            click_p2.play()
            next_p2 += ibi_p2

        keys = event.getKeys(keyList=[KEY_P1, KEY_P2, QUIT_KEY],
                             timeStamped=clock)
        for key, key_time in keys:
            if key == QUIT_KEY:
                quit_experiment(win, file_handles)
            elif key == KEY_P1:
                tap_data[1].append(key_time)
            elif key == KEY_P2:
                tap_data[2].append(key_time)

        msg.draw()
        win.flip()

    # Write taps
    beats_lookup = {1: beats_p1, 2: beats_p2}
    for p in [1, 2]:
        for tap_t in tap_data[p]:
            beat_t, async_ms = nearest_beat(tap_t, beats_lookup[p])
            sync_writer.writerow([
                dyad_id, condition, p,
                f'{tap_t:.4f}',
                f'{beat_t:.4f}' if not np.isnan(beat_t) else '',
                f'{async_ms:.2f}' if not np.isnan(async_ms) else '',
            ])

    show_message(
        f"Tapping task complete!\n\n"
        f"Press SPACE to continue."
    )


# =============================================================================
# TASK 4 — JOINT SIMON
# =============================================================================

def run_joint_simon(win, dyad_id, simon_writer, file_handles):
    clock = core.Clock()
    msg   = visual.TextStim(win, text='', color='white',
                            height=0.05, wrapWidth=0.9)

    def show_text(text, wait_key=True):
        msg.setText(text)
        msg.draw()
        win.flip()
        if wait_key:
            keys = event.waitKeys(keyList=['space', QUIT_KEY])
            if QUIT_KEY in keys:
                quit_experiment(win, file_handles)

    def run_block(block_num, trials, is_practice=False):
        for trial_num, trial in enumerate(trials, start=1):
            shape = trial['shape']
            side  = trial['side']
            pos   = STIM_LEFT_POS if side == 'left' else STIM_RIGHT_POS

            # P1 responds to square, P2 responds to diamond
            if shape == SHAPE_P1:
                expected_key = KEY_P1
                go_participant = 1
            else:
                expected_key = KEY_P2
                go_participant = 2
                
            # Flush anything pressed during the previous trial's feedback/ITI
            event.clearEvents()

            # Fixation
            draw_fixation(win)
            win.flip()
            core.wait(FIXATION_PRE)

            # Stimulus onset
            clock.reset()
            response_key = None
            rt_ms        = None
            responded    = False

            draw_fixation(win)
            if shape == 'square':
                draw_square(win, pos)
            else:
                draw_diamond(win, pos)
            win.flip()
            stim_onset = clock.getTime()

            while clock.getTime() - stim_onset < RESPONSE_WIN:
                t = clock.getTime() - stim_onset
                if t >= STIM_DUR and not responded:
                    draw_fixation(win)
                    win.flip()

                keys = event.getKeys(
                    keyList=[KEY_P1, KEY_P2, QUIT_KEY],
                    timeStamped=clock,
                )
                for k, kt in keys:
                    if k == QUIT_KEY:
                        quit_experiment(win, file_handles)
                    if not responded:
                        response_key = k
                        rt_ms        = (kt - stim_onset) * 1000
                        responded    = True
                        break
                if responded:
                    break

            # Flush any keys that arrived after the first response
            # (e.g. the other participant pressing their key just after
            # an error) so they don't bleed into the next trial.
            event.clearEvents()

            # Outcome
            if not responded:
                feedback = 'too_slow'
                correct  = False
            else:
                correct  = (response_key == expected_key)
                feedback = 'fixation' if correct else 'error'

            # Feedback
            if feedback == 'fixation':
                draw_fixation(win)
            else:
                msg.setText('Too slow' if feedback == 'too_slow' else 'Error')
                msg.draw()
            win.flip()
            core.wait(FEEDBACK_DUR)

            # ITI
            win.flip()
            core.wait(ITI_DUR)

            # Log both participants (skip practice)
            if not is_practice:
                for p in [1, 2]:
                    simon_writer.writerow([
                        dyad_id, trial_num, block_num, 'joint', p,
                        shape, side, response_key or '',
                        int(responded), int(correct),
                        f'{rt_ms:.2f}' if rt_ms is not None else '',
                        feedback,
                    ])

    # Instructions
    show_text(
        f"JOINT GO/NO-GO TASK\n\n"
        f"Participant 1  [ {KEY_P1.upper()} ]  → respond to  {SHAPE_P1.upper()}\n"
        f"Participant 2  [ {KEY_P2.upper()} ]  → respond to  {SHAPE_P2.upper()}\n\n"
        f"Only press your key for your shape. Ignore the other shape.\n\n"
        f"Press SPACE to begin the practice block."
    )

    run_block(0, build_simon_trial_list(PRACTICE_TRIALS), is_practice=True)

    show_text(
        f"Practice complete!\n\n"
        f"You have {EXP_BLOCKS} blocks of {EXP_TRIALS_PER_BLOCK} trials each.\n\n"
        f"Press SPACE to start Block 1."
    )

    for b in range(1, EXP_BLOCKS + 1):
        run_block(b, build_simon_trial_list(EXP_TRIALS_PER_BLOCK))
        if b < EXP_BLOCKS:
            show_text(
                f"Block {b} complete — take a short break.\n\n"
                f"Press SPACE when ready for Block {b + 1}."
            )
            win.flip()
            core.wait(PAUSE_BETWEEN_BLOCKS)

    show_text(
        "Joint task complete!\n\n"
        "Press SPACE to continue."
    )


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():

    # -------------------------------------------------------------------------
    # STEP 1 — Experimenter dialog: session info
    # -------------------------------------------------------------------------
    dlg = gui.Dlg(title='Session Setup')
    dlg.addField('Dyad ID:')
    dlg.addField('Synchrony condition:', choices=['synchrony', 'asynchrony'])
    dlg.show()
    if not dlg.OK:
        core.quit()

    dyad_id   = (dlg.data[0] or '').strip() or 'unknown'
    condition = dlg.data[1]

    # -------------------------------------------------------------------------
    # STEP 2 — Demographics dialogs (one per participant)
    # -------------------------------------------------------------------------
    demographics = {}   # {1: {age, gender}, 2: {age, gender}}

    for p in [1, 2]:
        dlg2 = gui.Dlg(title=f'Participant {p} — Demographics')
        dlg2.addField('Age:')
        dlg2.addField('Gender:', choices=['Female', 'Male', 'Other', 'Prefer not to say'])
        dlg2.show()
        if not dlg2.OK:
            core.quit()
        demographics[p] = {
            'age'   : (dlg2.data[0] or '').strip(),
            'gender': dlg2.data[1],
        }

    # -------------------------------------------------------------------------
    # STEP 3 — Output files
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    simon_path = os.path.join(OUTPUT_DIR, f'dyad{dyad_id}_simon_{timestamp}.csv')
    sync_path  = os.path.join(OUTPUT_DIR, f'dyad{dyad_id}_tapping_{timestamp}.csv')
    part_path  = os.path.join(OUTPUT_DIR, f'dyad{dyad_id}_group_info_{timestamp}.csv')

    file_handles = {}

    simon_fh = open(simon_path, 'w', newline='')
    sync_fh  = open(sync_path,  'w', newline='')
    part_fh  = open(part_path,  'w', newline='')
    file_handles['simon'] = simon_fh
    file_handles['sync']  = sync_fh
    file_handles['part']  = part_fh

    simon_writer = csv.writer(simon_fh)
    simon_writer.writerow([
        'dyad_id', 'trial', 'block', 'task', 'participant',
        'shape', 'side', 'response_key',
        'responded', 'correct', 'rt_ms', 'feedback',
    ])

    sync_writer = csv.writer(sync_fh)
    sync_writer.writerow([
        'dyad_id', 'condition', 'participant',
        'tap_time_s', 'beat_time_s', 'asynchrony_ms',
    ])

    part_writer = csv.writer(part_fh)
    part_writer.writerow([
        'dyad_id', 'participant', 'age', 'gender', 'consented',
    ])

    # -------------------------------------------------------------------------
    # STEP 4 — On-screen consent (shown to both participants together)
    # -------------------------------------------------------------------------
    win = make_window()

    msg = visual.TextStim(win, text='', color='white', height=0.04, wrapWidth=0.85)

    def show_text(text, wait_key=True):
        msg.setText(text)
        msg.draw()
        win.flip()
        if wait_key:
            keys = event.waitKeys(keyList=['space', QUIT_KEY])
            if QUIT_KEY in keys:
                quit_experiment(win, file_handles)

    CONSENT_TEXT = (
        "You are invited to participate in our SocCult exam.\n\n"
        "Participation is voluntary. You may withdraw at any time without consequence.\n\n"
        "No names are collected. You are identified only by a random Dyad ID, and data cannot be traced back to you.\n\n"
        "Questions? Please ask before continuing.\n\n"
        "By pressing SPACE you confirm that you BOTH consent to participate.\n\n"
        "Press SPACE to consent or ESCAPE to withdraw."
    )

    show_text(CONSENT_TEXT, wait_key=True)

    # Record consent for both participants
    for p in [1, 2]:
        part_writer.writerow([
            dyad_id, p,
            demographics[p]['age'],
            demographics[p]['gender'],
            'yes',
        ])
    part_fh.flush()

    # -------------------------------------------------------------------------
    # STEP 5 — Welcome / overview
    # -------------------------------------------------------------------------
    show_text(
        f"Thank you, we have recorded your consent.\n\n"
        f"Participant 1  →  key  [ {KEY_P1.upper()} ]\n"
        f"Participant 2  →  key  [ {KEY_P2.upper()} ]\n\n"
        f"The session has four parts:\n"
        f"  1. Individual Simon Task for Participant 1\n"
        f"  2. Individual Simon Task for Participant 2\n"
        f"  3. Synchrony Tapping Task together\n"
        f"  4. Joint Simon Task\n\n"
        f"Press SPACE to begin."
    )

    # =========================================================================
    # PART 1 — Individual Simon, Participant 1
    # =========================================================================
    show_text(
        "PART 1 OF 4\n\n"
        "Participant 1, please sit at the keyboard.\n"
        "\n"
        "Participant 2, please wait.\n\n"
        "\n"
        "Press SPACE when ready."
    )
    run_individual_simon(win, participant=1, dyad_id=dyad_id,
                         simon_writer=simon_writer, file_handles=file_handles)

    # =========================================================================
    # PART 2 — Individual Simon, Participant 2
    # =========================================================================
    show_text(
        "PART 2 OF 4\n\n"
        "Participant 2, please sit at the keyboard.\n"
        "\n"
        "Participant 1, please wait.\n\n"
        "\n"
        "Press SPACE when ready."
    )
    run_individual_simon(win, participant=2, dyad_id=dyad_id,
                         simon_writer=simon_writer, file_handles=file_handles)

    # =========================================================================
    # PART 3 — Synchrony / Asynchrony Tapping
    # =========================================================================
    show_text(
        "PART 3 OF 4\n\n"
        "Both participants, please sit together at the keyboard.\n\n"
        "\n"
        "Press SPACE when ready."
    )
    run_synchrony_task(win, condition, dyad_id=dyad_id,
                       sync_writer=sync_writer, file_handles=file_handles)

    # =========================================================================
    # PART 4 — Joint Simon
    # =========================================================================
    show_text(
        "PART 4 OF 4\n\n"
        "Both participants remain at the keyboard.\n\n"
        "\n"
        "Press SPACE when ready."
    )
    run_joint_simon(win, dyad_id=dyad_id, simon_writer=simon_writer, file_handles=file_handles)

    # -- Done -----------------------------------------------------------------
    show_text(
        "Thank you for participating!\n\nThe experiment will now end...\n\n But our love for you will not! ♥\n\n"
        "Press SPACE to exit."
    )

    for fh in file_handles.values():
        fh.close()
    win.close()
    core.quit()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    run_experiment()
