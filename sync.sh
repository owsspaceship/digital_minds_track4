#!/usr/bin/env bash
# Run this from the parent directory of digital_minds_track4 (e.g. ~/)
# after downloading a fresh zip into ~/Downloads.
#
# Usage: bash sync.sh
set -e

cd ~/Downloads
ZIP=$(ls -t digital_minds_track4*.zip | head -1)
echo "Using zip: $ZIP"

cd ~
rm -rf digital_minds_track4
unzip -q "Downloads/$ZIP" -d digital_minds_track4
cd digital_minds_track4

git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/owsspaceship/digital_minds_track4.git

BRANCH=$(git branch --show-current)
git push -u origin "$BRANCH" --force

echo ""
echo "Synced and pushed as branch: $BRANCH"
echo "Next: export ANTHROPIC_API_KEY=your_key && python check_api.py"
