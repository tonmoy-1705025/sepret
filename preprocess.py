import argparse
import json
import os
import librosa
from tqdm import tqdm


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):

    file_infos = []
    in_dir = os.path.abspath(in_dir)                        # Return absolute path
    wav_list = os.listdir(in_dir)                           # Return to the list of files in this directory

    for wav_file in tqdm(wav_list, desc="Sound Processing.."):
        if not wav_file.endswith('.wav'):                   # Determine whether it ends with .wav
            continue
        wav_path = os.path.join(in_dir, wav_file)           # splicing path
        samples, _ = librosa.load(wav_path, sr=sample_rate) # Read voice
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):                         # If the output path does not exist, create it
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)                  # Write information to json


def preprocess(args):
    
    
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(args['in_dir'], data_type, speaker),  # splicing path
                               os.path.join(args['out_dir'], data_type),
                               speaker,
                               sample_rate=args['sample_rate'])
            
            # print("=============================================")
            # print(os.path.join(args.in_dir, data_type, speaker))
            # print(os.path.join(args.out_dir, data_type))
            # print(speaker)
            # print(args.sample_rate)
            # exit()

preprocess_params = {
    'in_dir': "./min/",
    'out_dir': "./json/",
    'sample_rate' : 8000
}
            
if __name__ == "__main__":
    preprocess(preprocess_params)
