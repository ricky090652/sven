#!/bin/bash
# BigCodeBench Results Viewer
# Usage: ./view_results.sh [output_name]

OUTPUT_DIR="../experiments/bigcodebench"
OUTPUT_NAME="${1:-qwen-1.5b-docker}"

echo "======================================"
echo "BigCodeBench Evaluation Results"
echo "======================================"
echo ""

# Find result files
PASS_K_FILE=$(find "$OUTPUT_DIR" -name "${OUTPUT_NAME}*pass_at_k.json" 2>/dev/null | head -1)
EVAL_FILE=$(find "$OUTPUT_DIR" -name "${OUTPUT_NAME}*eval_results.json" 2>/dev/null | head -1)
SAMPLES_FILE=$(find "$OUTPUT_DIR" -name "${OUTPUT_NAME}*.jsonl" ! -name "*sanitized*" 2>/dev/null | head -1)

# Check if files exist
if [ -z "$PASS_K_FILE" ]; then
    echo "❌ Pass@k results not found!"
    echo ""
    echo "Available files in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR" | grep "$OUTPUT_NAME" || echo "  No files found with name: $OUTPUT_NAME"
    echo ""
    echo "💡 The evaluation might still be running or failed."
    echo "   Check running processes with: ps aux | grep bigcodebench"
    exit 1
fi

echo "📁 Result Files:"
echo "  Pass@k:    $PASS_K_FILE"
echo "  Eval:      $EVAL_FILE"
echo "  Samples:   $SAMPLES_FILE"
echo ""

# Display Pass@k results
echo "======================================"
echo "📊 Pass@k Scores"
echo "======================================"
if command -v jq &> /dev/null; then
    cat "$PASS_K_FILE" | jq '.'
else
    cat "$PASS_K_FILE"
fi
echo ""

# Summary statistics if eval file exists
if [ -n "$EVAL_FILE" ] && [ -f "$EVAL_FILE" ]; then
    echo "======================================"
    echo "📈 Summary Statistics"
    echo "======================================"
    
    if command -v jq &> /dev/null; then
        TOTAL=$(cat "$EVAL_FILE" | jq '.eval | length')
        PASSED=$(cat "$EVAL_FILE" | jq '[.eval | to_entries[] | select(.value.status == "pass")] | length')
        FAILED=$(cat "$EVAL_FILE" | jq '[.eval | to_entries[] | select(.value.status == "fail")] | length')
        
        echo "Total tasks:   $TOTAL"
        echo "Passed:        $PASSED"
        echo "Failed:        $FAILED"
        echo "Pass rate:     $(awk "BEGIN {printf \"%.2f%%\", ($PASSED/$TOTAL)*100}")"
    else
        echo "Install 'jq' for detailed statistics: sudo apt-get install jq"
    fi
    echo ""
fi

# Sample count
if [ -n "$SAMPLES_FILE" ] && [ -f "$SAMPLES_FILE" ]; then
    SAMPLE_COUNT=$(wc -l < "$SAMPLES_FILE")
    echo "Generated samples: $SAMPLE_COUNT"
fi

echo ""
echo "======================================"
echo "To view detailed results:"
echo "  cat $EVAL_FILE | jq '.eval.\"BigCodeBench/0\"'"
echo "======================================"
