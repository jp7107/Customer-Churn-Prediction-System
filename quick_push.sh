#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 GITHUB PUSH HELPER - Follow These Steps"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "STEP 1: Create GitHub Repository"
echo "  → Go to: https://github.com/new"
echo "  → Name: telco-churn-prediction (or your choice)"
echo "  → Make it PUBLIC ⚠️"
echo "  → DO NOT initialize with README"
echo "  → Click 'Create repository'"
echo ""
echo "STEP 2: Copy Your Repository URL"
echo "  It will look like:"
echo "  https://github.com/YOUR_USERNAME/telco-churn-prediction.git"
echo ""
read -p "Press ENTER when you've created the repo and copied the URL..."
echo ""
read -p "Paste your repository URL here: " REPO_URL
echo ""

# Validate URL
if [[ -z "$REPO_URL" ]]; then
    echo "❌ ERROR: No URL provided. Please try again."
    exit 1
fi

echo "STEP 3: Connecting and Pushing..."
echo ""

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    echo "⚠️  Remote 'origin' already exists. Updating it..."
    git remote set-url origin "$REPO_URL"
else
    echo "Adding remote 'origin'..."
    git remote add origin "$REPO_URL"
fi

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin master 2>&1 | tee /tmp/git_push_log.txt

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ SUCCESS! Your project is on GitHub!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Your repository URL:"
    echo "  ${REPO_URL%.git}"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "  1. ✅ GitHub push (DONE!)"
    echo "  2. ⏳ Record 5-min demo video"
    echo "  3. ⏳ Upload video to Google Drive"
    echo "  4. ⏳ Email submission to: info@cognirexsolutions.in"
    echo ""
    echo "See GITHUB_PUSH_GUIDE.md for detailed email template."
    echo ""
else
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ❌ PUSH FAILED - Common Solutions"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    cat /tmp/git_push_log.txt
    echo ""
    echo "Common fixes:"
    echo "  1. Wrong URL? Remove and re-add:"
    echo "     git remote remove origin"
    echo "     git remote add origin YOUR_CORRECT_URL"
    echo ""
    echo "  2. Authentication issue? Use HTTPS and GitHub personal token"
    echo ""
    echo "  3. Branch name issue? Try:"
    echo "     git branch -M main"
    echo "     git push -u origin main"
    echo ""
fi
