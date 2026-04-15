#!/bin/bash
# MemoryKernel Terminal Workflow Example
# =======================================
# This script demonstrates a typical terminal workflow with memk

set -e

echo "🧠 MemoryKernel Terminal Workflow Demo"
echo "======================================"
echo

# 1. Initialize workspace
echo "📁 Step 1: Initialize workspace..."
memk init
echo "✅ Workspace initialized"
echo

# 2. Ingest Git history
echo "📚 Step 2: Ingest Git history..."
memk ingest git --limit 20
echo "✅ Git history ingested"
echo

# 3. Add manual knowledge
echo "💭 Step 3: Add manual knowledge..."
memk remember "The main API endpoint is /api/v1"
memk remember "Database connection string is in .env file"
memk remember "Tests are run with: pytest tests/"
echo "✅ Manual knowledge added"
echo

# 4. Search for information
echo "🔍 Step 4: Search for information..."
echo "Searching for 'API'..."
memk search "API"
echo

# 5. Build context
echo "📝 Step 5: Build context for a query..."
echo "Building context for: 'How do I run tests?'"
memk context "How do I run tests?"
echo

# 6. Check system health
echo "🏥 Step 6: Check system health..."
memk doctor
echo

# 7. Start file watcher (optional)
echo "👀 Step 7: Start file watcher..."
memk watch start
echo "✅ File watcher started"
echo

echo "🎉 Demo complete!"
echo
echo "Try these commands yourself:"
echo "  memk remember 'your knowledge here'"
echo "  memk search 'your query'"
echo "  memk context 'your question'"
echo "  memk doctor"
