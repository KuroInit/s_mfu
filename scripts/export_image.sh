#!/bin/bash
# Build the Docker image and export it as a tar.gz for transfer to NSCC.
# On NSCC, Singularity can run the resulting archive directly.
#
# Usage: bash scripts/export_image.sh [output_name]
# Example: bash scripts/export_image.sh moe-cap-sweep

set -euo pipefail

IMAGE_TAG="moe-cap-sweep:latest"
OUTPUT_NAME="${1:-moe-cap-sweep}"
OUTPUT_FILE="${OUTPUT_NAME}.tar.gz"

echo "Building Docker image: $IMAGE_TAG"
docker build -t "$IMAGE_TAG" .

echo "Exporting to $OUTPUT_FILE ..."
docker save "$IMAGE_TAG" | gzip > "$OUTPUT_FILE"

SIZE=$(du -sh "$OUTPUT_FILE" | cut -f1)
echo ""
echo "Done: $OUTPUT_FILE  ($SIZE)"
echo ""
echo "Transfer to NSCC:"
echo "  scp $OUTPUT_FILE <user>@aspire.nscc.sg:/scratch/<project>/"
echo ""
echo "On NSCC, convert to Singularity image:"
echo "  singularity build ${OUTPUT_NAME}.sif docker-archive://${OUTPUT_FILE}"
echo ""
echo "Run on NSCC via PBS:"
echo "  singularity exec --nv ${OUTPUT_NAME}.sif bash /workspace/scripts/run_single.sh <model> <bs> <config>"
