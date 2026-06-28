set -e
VPY=/home/ian/.venvs/venv/bin/python
echo "### C2 TRAIN ###"
$VPY scripts/run_train.py
echo "### COPY CKPT ###"
cp outputs/train/sequence_vae_full.pt models/sequence_vae/sequence_vae_full.pt
ls -la models/sequence_vae/
echo "### C2 STRESS TEST ###"
$VPY scripts/run_stress_test.py
echo "### C2 CALIBRATE ###"
$VPY scripts/run_calibrate.py
echo "### C2 SCORE ###"
$VPY scripts/run_score.py
echo "### C2 CHAIN COMPLETE ###"
