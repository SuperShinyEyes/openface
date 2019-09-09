#!/bin/bash
video_id=8

echo "%07d" 1

# ffmpeg -framerate 24 -i "results-aligned${video_id}/unravel${video_id}_frame%d_aligned.png" -vb 20M "unravel${video_id}_aligned.mp4"

ffmpeg -framerate 24 -i %d.png -vb 20M unravel8_aligned.mp4
# ffmpeg -framerate 24 -i results-aligned8/*.png -vb 20M unravel8_aligned.mp4
# ffmpeg -framerate 24 -i "*.png" -vb 20M "unravel${video_id}_aligned.mp4"