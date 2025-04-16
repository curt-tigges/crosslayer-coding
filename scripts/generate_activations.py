import argparse
import sys
import os
import json  # For potential future use with dict args

# Ensure the clt package is discoverable
# This assumes the script is run from the root of the project
# Or that the clt package is installed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from clt.activation_generation.generator import ActivationGenerator
from clt.config.data_config import ActivationConfig  # Import ActivationConfig


def parse_arguments():
    """Parse command-line arguments for activation generation."""
    parser = argparse.ArgumentParser(
        description="Generate and save model activations using ActivationConfig."
    )

    # Arguments map directly to ActivationConfig fields
    # --- Model Source ---
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name or path."
    )
    parser.add_argument(
        "--mlp_input_template",
        type=str,
        required=True,
        help="NNsight path template for MLP inputs.",
    )
    parser.add_argument(
        "--mlp_output_template",
        type=str,
        required=True,
        help="NNsight path template for MLP outputs.",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default=None,
        help="Optional model dtype (e.g., 'float16').",
    )

    # --- Dataset Source ---
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Dataset name or path."
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split."
    )
    parser.add_argument(
        "--dataset_text_column",
        type=str,
        default="text",
        help="Dataset text column name.",
    )

    # --- Generation Parameters ---
    parser.add_argument(
        "--context_size",
        type=int,
        default=128,
        help="Context size for tokenization/inference.",
    )
    parser.add_argument(
        "--inference_batch_size", type=int, default=512, help="Inference batch size."
    )
    parser.add_argument(
        "--exclude_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude special tokens.",
    )
    parser.add_argument(
        "--prepend_bos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prepend BOS token.",
    )

    # --- Dataset Handling ---
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF dataset streaming.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust remote code for dataset.",
    )
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Optional HF dataset cache path."
    )

    # --- Generation Output Control ---
    parser.add_argument(
        "--target_total_tokens",
        type=int,
        default=None,
        help="Target number of tokens to generate.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max dataset samples to process."
    )

    # --- Storage Parameters ---
    parser.add_argument(
        "--activation_dir",
        type=str,
        default="./activations",
        help="Base directory to save activations.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="hdf5",
        choices=["hdf5", "npz"],
        help="Activation file format.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lz4",
        help="Compression ('lz4', 'gzip', or 'None').",
    )
    parser.add_argument(
        "--chunk_size_tokens",
        type=int,
        default=1_000_000,
        help="Target tokens per chunk file.",
    )

    # --- Normalization ---
    parser.add_argument(
        "--compute_norm_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and save normalization stats.",
    )

    # --- Workflow / Execution Parameters (Not part of ActivationConfig intrinsically) ---
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for generation ('cuda', 'cpu').",
    )
    parser.add_argument(
        "--storage_type",
        type=str,
        default="local",
        choices=["local", "remote"],
        help="Storage workflow type ('local' saves files, 'remote' prepares for server but is stubbed).",
    )

    # --- NNsight Arguments (Potentially pass as JSON strings) ---
    parser.add_argument(
        "--nnsight_tracer_kwargs_json",
        type=str,
        default="{}",
        help="JSON string for nnsight model.trace() kwargs.",
    )
    parser.add_argument(
        "--nnsight_invoker_args_json",
        type=str,
        default="{}",
        help="JSON string for nnsight model.trace() invoker_args.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Handle potential None for compression argument
    if args.compression and args.compression.lower() == "none":
        compression_algo = None
    else:
        compression_algo = args.compression

    # Parse JSON strings for NNsight kwargs
    try:
        nnsight_tracer_kwargs = json.loads(args.nnsight_tracer_kwargs_json)
        nnsight_invoker_args = json.loads(args.nnsight_invoker_args_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding NNsight JSON arguments: {e}")
        print("Please provide valid JSON strings or empty dicts '{}'.")
        sys.exit(1)

    # Create ActivationConfig from parsed arguments
    activation_config = ActivationConfig(
        model_name=args.model_name,
        mlp_input_module_path_template=args.mlp_input_template,
        mlp_output_module_path_template=args.mlp_output_template,
        model_dtype=args.model_dtype,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
        context_size=args.context_size,
        inference_batch_size=args.inference_batch_size,
        exclude_special_tokens=args.exclude_special_tokens,
        prepend_bos=args.prepend_bos,
        streaming=args.streaming,
        dataset_trust_remote_code=args.trust_remote_code,
        cache_path=args.cache_path,
        target_total_tokens=args.target_total_tokens,
        max_samples=args.max_samples,
        activation_dir=args.activation_dir,
        output_format=args.output_format,
        compression=compression_algo,
        chunk_size_tokens=args.chunk_size_tokens,
        compute_norm_stats=args.compute_norm_stats,
        nnsight_tracer_kwargs=nnsight_tracer_kwargs,
        nnsight_invoker_args=nnsight_invoker_args,
    )

    # Instantiate the generator, passing the config and optional device override
    generator = ActivationGenerator(
        activation_config=activation_config,
        device=args.device,  # Pass device separately
    )

    # Set the storage type explicitly (controls workflow, not part of ActivationConfig)
    generator.set_storage_type(args.storage_type)

    # Run the generation process - it now uses the config internally
    generator.generate_and_save()

    print("Activation generation script finished.")


if __name__ == "__main__":
    main()
