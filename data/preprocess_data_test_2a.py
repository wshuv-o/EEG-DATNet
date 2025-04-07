# import mne
# import numpy as np
# import scipy.io
# import os

# factor_new = 1e-3
# init_block_size = 1000

# data_path = 'D:/EEG-TransNet/data2/gdf'
# data_files = ['A0'+str(i)+'E.gdf' for i in range(1, 10)]

# label_path = 'D:/EEG-TransNet/data2/a'
# label_files = ['A0'+str(i)+'E.mat' for i in range(1, 10)]

# save_path = 'dataset/bci_iv_2a'

# event_description = {'783': "CueUnknown"}

# # for file in data_files:
# #     # Read raw data

# #     raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)

# #     # Get events from annotations
# #     raw_events, all_event_id = mne.events_from_annotations(raw_data)

# #     # Manually adjust event IDs and convert to uint16
# #     raw_events[:, 2] = raw_events[:, 2].astype(np.uint16)  # Ensure event IDs are uint16

# #     # Process data further (converting to uV)
# #     raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)

# #     # Mark bad channels
# #     raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

# #     # Pick EEG channels
# #     test_picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

# #     # Time window for epochs
# #     tmin, tmax = 0, 4

# #     # Create event_id dictionary from event_description
# #     event_id = dict()
# #     for event in all_event_id:
# #         if event in event_description:
# #             event_id[event_description[event]] = all_event_id[event]

# #     # Create epochs from raw data
# #     raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=test_picks, baseline=None, preload=True)

# #     # Get the data and labels
# #     data = raw_epochs.get_data()  # [n_epochs, n_channels, n_times]
# #     data = data[:, :, :-1]  # Remove last time point if needed

# #     # Save data to numpy files
# #     np.save(os.path.join(save_path, file[:-4] + '_data.npy'), data)

# for file in label_files:
#     true_label = scipy.io.loadmat(os.path.join(label_path, file))
#     label = true_label['classlabel']
#     np.save(os.path.join(save_path, file[:-4] + '_label.npy'), label)
