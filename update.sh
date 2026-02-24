#!/bin/bash
# ═══════════════════════════════════════════════════
# ExerciseAI — Quick Update Script
# Use this to push new versions to GitHub
#
# USAGE:
#   ./update.sh "v15.1 — added new feature"
#   ./update.sh   (uses default message)
# ═══════════════════════════════════════════════════

cd ~/Documents/ExerciseAI || { echo "❌ Project not found. Run setup_github.sh first."; exit 1; }

# Check if a new HTML file exists in Downloads
NEW_FILE=$(ls -t ~/Downloads/ExerciseAI*.html 2>/dev/null | head -1)
if [ -n "$NEW_FILE" ]; then
    cp "$NEW_FILE" ./index.html
    echo "✅ Updated index.html from: $(basename "$NEW_FILE")"
else
    echo "ⓘ No new ExerciseAI HTML found in Downloads — pushing current state"
fi

# Commit message
MSG="${1:-Update ExerciseAI}"

git add .
git commit -m "$MSG" || { echo "ⓘ Nothing to commit."; exit 0; }
git push

echo ""
echo "✅ Pushed to GitHub: $MSG"
echo "   Live site will update in ~1 minute."
