#!/usr/bin/env python3
#
# Example Command:
# python3 create_call_image.py --input_dir ./my_audio_calls --output_dir ./spectrograms --n_mels 128 --n_fft 2048 --hop_length 512
#

import argparse
import os
from pathlib import Path
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from PIL import Image
from tqdm import tqdm
import numpy as np


def create_spectrogram(
        waveform,
        sample_rate,
        target_sample_rate,
        n_fft,
        hop_length,
        n_mels
):
    """
    Transforms a waveform into a log-mel spectrogram image tensor.
    """
    # 1. Resample if the sample rate is not the target rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # 2. Define the spectrogram transforms
    # MelSpectrogram creates a spectrogram on the Mel scale, which is better for human/animal perception
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=target_sample_rate // 2
    ).to(waveform.device)

    # AmplitudeToDB converts the linear amplitude to a decibel (logarithmic) scale
    amplitude_to_db_transform = AmplitudeToDB(stype='power', top_db=80).to(waveform.device)

    # 3. Apply the transforms
    mel_spec = mel_spectrogram_transform(waveform)
    mel_spec_db = amplitude_to_db_transform(mel_spec)

    return mel_spec_db


def normalize_and_save_image(spec_tensor, output_path):
    """
    Normalizes a spectrogram tensor and saves it as a grayscale PNG image.
    """
    # Normalize the tensor to the range [0, 1]
    min_val = spec_tensor.min()
    max_val = spec_tensor.max()
    if max_val > min_val:
        normalized_spec = (spec_tensor - min_val) / (max_val - min_val)
    else:
        # Handle the case where the spectrogram is flat (e.g., pure silence)
        normalized_spec = torch.zeros_like(spec_tensor)

    # Scale to [0, 255] and convert to an 8-bit integer type
    image_tensor = (normalized_spec * 255).byte()

    # Squeeze to remove the channel dimension (e.g., from [1, H, W] to [H, W])
    image_tensor = image_tensor.squeeze()

    # Convert to a NumPy array on the CPU
    numpy_array = image_tensor.cpu().numpy()

    # Create a PIL Image from the NumPy array
    # Spectrograms are usually displayed with low frequencies at the bottom,
    # but the array has them at the top, so we flip it vertically.
    image = Image.fromarray(numpy_array, mode='L')  # 'L' is for grayscale
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image
    image.save(output_path)


def main():
    """
    Main function to parse arguments and process audio files.
    """
    parser = argparse.ArgumentParser(
        description="Create spectrogram images from audio files using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Renamed --working_dir to --input_dir for clarity
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the input audio files (.wav, .mp3, .flac)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory where the output spectrogram images will be saved."
    )

    # Important parameters for spectrogram generation
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help="Target sample rate to resample all audio files to."
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=2048,
        help="Size of the Fast Fourier Transform (FFT) window."
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=512,
        help="Number of samples between successive FFT windows."
    )
    parser.add_argument(
        '--n_mels',
        type=int,
        default=128,
        help="Number of Mel frequency bands."
    )

    args = parser.parse_args()

    # --- Setup and Validation ---
    print("--- Configuration ---")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Sample Rate: {args.sample_rate}")
    print(f"FFT Size (n_fft): {args.n_fft}")
    print(f"Hop Length: {args.hop_length}")
    print(f"Mel Bands (n_mels): {args.n_mels}")
    print("---------------------\n")

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # [DEBUG] Check if input directory exists
    if not input_path.is_dir():
        print(f"[DEBUG] ERROR: Input directory not found at '{input_path}'")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: '{output_path.resolve()}'")

    # Find all supported audio files recursively
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = [f for f in input_path.rglob('*') if f.suffix.lower() in audio_extensions]

    # [DEBUG] Check if any audio files were found
    if not audio_files:
        print(f"[DEBUG] ERROR: No audio files with extensions {audio_extensions} found in '{input_path}'")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    # --- Processing Loop ---
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # 1. Load audio file
            waveform, sr = torchaudio.load(audio_file)

            # Convert to mono by averaging channels if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 2. Create spectrogram
            spec_tensor = create_spectrogram(
                waveform,
                sample_rate=sr,
                target_sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels
            )

            # 3. Define output path and save the image
            output_filename = output_path / f"{audio_file.stem}.png"
            normalize_and_save_image(spec_tensor, output_filename)

        except Exception as e:
            # [DEBUG] Print a helpful message if a file fails
            print(f"\n[DEBUG] FAILED to process {audio_file.name}. Reason: {e}")
            # Continue to the next file
            continue

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()