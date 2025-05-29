"""
Compress an existing HDF5 activation dataset into Zarr format with Blosc compression.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import zarr
from tqdm import tqdm

# Ensure the clt package is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import CLT-specific modules if needed later, e.g., for metadata validation
# For now, direct data manipulation might suffice.

logger = None  # Initialize logger, will be configured in main


def setup_logging():
    """Sets up basic logging."""
    global logger
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    """Parse command-line arguments for dataset compression."""
    parser = argparse.ArgumentParser(description="Compress an HDF5 activation dataset to Zarr format.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input HDF5 activation dataset directory (e.g., ./activations/model/dataset_split).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output Zarr activation dataset directory.",
    )
    parser.add_argument(
        "--zarr-chunk-size",
        type=int,
        default=1000,  # Default Zarr chunking along the token dimension
        help="Chunk size for the first dimension (tokens) in the output Zarr arrays.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,  # Blosc default is 5
        help="Blosc compression level (0-9). 0 means no compression, 9 is max.",
    )
    parser.add_argument(
        "--compression-shuffle",
        type=int,
        default=1,  # Default to Blosc.SHUFFLE
        choices=[0, 1, 2],  # Blosc.NOSHUFFLE, Blosc.SHUFFLE, Blosc.BITSHUFFLE
        help="Blosc shuffle type (0: None, 1: Byte Shuffle, 2: Bit Shuffle). Default is 1 (Byte).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,  # Defaults to number of cores if None
        help="Number of worker threads for parallel chunk processing. Defaults to os.cpu_count().",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_arguments()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.is_dir():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)

    if output_path.exists():
        logger.warning(f"Output directory {output_path} already exists. It might be overwritten.")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting compression from HDF5 ({input_path}) to Zarr ({output_path})")
    logger.info(f"Zarr chunk size (tokens): {args.zarr_chunk_size}")
    logger.info(f"Blosc compression level: {args.compression_level}, shuffle: {args.compression_shuffle}")

    # 1. Copy metadata files
    files_to_copy = ["metadata.json", "norm_stats.json", "index.bin"]
    for file_name in files_to_copy:
        src_file = input_path / file_name
        dst_file = output_path / file_name
        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {file_name} to {output_path}")
            except Exception as e:
                logger.error(f"Failed to copy {src_file} to {dst_file}: {e}")
        else:
            logger.warning(f"Metadata file {src_file} not found, skipping copy.")

    # 2. Discover HDF5 chunk files
    hdf5_chunks = sorted(input_path.glob("chunk_*.h5"))
    if not hdf5_chunks:
        logger.error(f"No HDF5 chunk files (chunk_*.h5) found in {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(hdf5_chunks)} HDF5 chunks to process.")

    # Determine Blosc shuffle type
    if args.compression_shuffle == 0:
        shuffle_type = zarr.Blosc.NOSHUFFLE
    elif args.compression_shuffle == 2:
        shuffle_type = zarr.Blosc.BITSHUFFLE
    else:  # Default to 1
        shuffle_type = zarr.Blosc.SHUFFLE

    compressor = zarr.Blosc(cname="lz4", clevel=args.compression_level, shuffle=shuffle_type)

    # 3. Process each HDF5 chunk and convert to Zarr
    # For now, sequential processing. Parallelism can be added later.
    for hdf5_file_path in tqdm(hdf5_chunks, desc="Compressing chunks"):
        chunk_name = hdf5_file_path.stem  # e.g., "chunk_0"
        zarr_chunk_group_path = output_path / chunk_name

        try:
            with h5py.File(hdf5_file_path, "r") as hf_in:
                # Create a Zarr group for this chunk
                # Using DirectoryStore directly for each chunk group
                store = zarr.DirectoryStore(str(zarr_chunk_group_path))
                z_chunk_group = zarr.group(store=store, overwrite=True)

                for layer_key in hf_in.keys():  # e.g., "layer_0", "layer_1"
                    if not layer_key.startswith("layer_"):
                        logger.warning(f"Skipping non-layer group '{layer_key}' in {hdf5_file_path}")
                        continue

                    hf_layer_group = hf_in[layer_key]
                    z_layer_group = z_chunk_group.create_group(layer_key, overwrite=True)

                    for dataset_name in hf_layer_group.keys():  # "inputs", "targets"
                        hf_dataset = hf_layer_group[dataset_name]
                        data_shape = hf_dataset.shape
                        data_dtype = hf_dataset.dtype

                        # Define Zarr array chunks - chunk along tokens, keep feature dim intact
                        zarr_chunks_shape = (min(args.zarr_chunk_size, data_shape[0]), data_shape[1])

                        # Create Zarr array
                        z_array = z_layer_group.create_dataset(
                            dataset_name,
                            shape=data_shape,
                            chunks=zarr_chunks_shape,
                            dtype=data_dtype,
                            compressor=compressor,
                            overwrite=True,
                        )

                        # Copy data from HDF5 to Zarr
                        # This will read the entire HDF5 dataset into memory if not chunked properly during read.
                        # For very large datasets, chunked reading from HDF5 might be needed.
                        # However, individual HDF5 datasets per layer/type are usually manageable.
                        z_array[:] = hf_dataset[:]

            logger.debug(f"Successfully converted {hdf5_file_path} to Zarr group at {zarr_chunk_group_path}")

        except Exception as e:
            logger.error(f"Error processing chunk {hdf5_file_path}: {e}")
            logger.error(f"Skipping this chunk. The output dataset at {output_path} might be incomplete.")
            # Optionally, clean up partially written zarr_chunk_group_path
            if zarr_chunk_group_path.exists():
                try:
                    shutil.rmtree(zarr_chunk_group_path)
                except Exception as rm_e:
                    logger.error(f"Failed to clean up partially written Zarr chunk {zarr_chunk_group_path}: {rm_e}")
            continue  # Move to the next chunk

    logger.info("Dataset compression to Zarr format complete.")
    logger.info(f"Output Zarr dataset saved to: {output_path}")
    logger.warning(
        "Important: The LocalActivationStore does not currently support Zarr. "
        "You will need to modify it or use a different store to read this Zarr dataset."
    )


if __name__ == "__main__":
    main()
