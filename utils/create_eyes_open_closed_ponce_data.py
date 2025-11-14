import numpy as np
import pandas as pd
import h5py, os

dates = ['250225', '260225']
monkey_name = 'D'
sampling_rate = 1000  # Hz, original
# go up two levels from the utils directory
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(current_dir))
print(main_dir)
for date in dates:
    threshold = -0.5
    file_path = f'{main_dir}/data/ponce/{monkey_name}/{date}/RS'
    mua_darkness_file = [f for f in os.listdir(file_path) if f.endswith('.mat')][0]
    f = h5py.File(os.path.join(file_path, mua_darkness_file), 'r')
    eyeChans = f['eyeChans'][:]

    pupil = eyeChans[:, 2]
    pupil = pupil - pupil.min()  # Chen-style

    # Downsample to 1Hz
    trimmed = pupil[:len(pupil) // 1000 * 1000]  # ensure divisible
    downsampled = trimmed.reshape(-1, 1000).mean(axis=1)  # shape: (T,)

    # Behavioral state: pupil > threshold
    state = (downsampled > threshold).astype(float)

    # Smooth with 3-point moving average
    kernel = np.ones(3) / 3
    smoothed_state = np.convolve(state, kernel, mode='same')
    smoothed_state = (smoothed_state >= 0.5).astype(int)

    # Epoch extraction
    def get_mean_state(vals):
        return 'Closed_eyes' if np.sum(vals == 0) > np.sum(vals == 1) else 'Open_eyes'

    changes = np.where(np.diff(smoothed_state) != 0)[0]
    edges = [0] + changes.tolist() + [len(smoothed_state)]
    i_start, i_stop, states = [edges[0]], [edges[1]], [get_mean_state(smoothed_state[edges[0]:edges[1]])]
    for start, stop in zip(edges[1:-1], edges[2:]):
        next_state = get_mean_state(smoothed_state[start:stop])
        if next_state == states[-1]:
            i_stop[-1] = stop
        else:
            i_start.append(start)
            i_stop.append(stop)
            states.append(next_state)

    # Final timing info
    start_times = np.array(i_start).astype(float)  # already in seconds (1 Hz)
    stop_times = np.array(i_stop).astype(float)
    durs = stop_times - start_times

    # DataFrame
    epochs = pd.DataFrame({
        't_start': start_times,
        't_stop': stop_times,
        'dur': durs,
        'state': states
    })

    # Save
    save_path = f'{main_dir}/data/ponce/metadata/{monkey_name}/epochs_{monkey_name}_RS_{date}.csv'
    epochs.to_csv(save_path, index=False)
