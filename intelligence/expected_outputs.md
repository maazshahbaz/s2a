# Expected Outputs from Customer Support Transcription Processing

## What to Expect

When you process your customer support transcriptions with the Action Pipeline, here's what you should expect based on the sample transcription you provided:

### 1. For Each Transcription Record

**Input Example** (from your JSON):
- Customer calling about PC issues
- Cannot download anything or install games
- Windows activation problems
- AnyDesk remote support session
- Technical troubleshooting steps

**Expected Output Structure**:
```json
{
  "success": true,
  "data": {
    "summary": "Customer support call about PC download and Windows activation issues requiring remote assistance",
    "action_items": [
      {
        "assignee": "Support technician",
        "task": "Replace SSD/NVME drive and fresh install Windows",
        "due_date": "2024-06-17",
        "priority": "high",
        "confidence": 0.9,
        "span": "We're going to send you a replacement for that. We're going to fresh install Windows on it"
      },
      {
        "assignee": "Customer",
        "task": "Call back tomorrow morning",
        "due_date": "2024-06-17", 
        "priority": "medium",
        "confidence": 0.8,
        "span": "Well, give me a call tomorrow. Yeah, tomorrow morning will be better"
      }
    ],
    "entities": {
      "invoice_ids": [],
      "order_ids": [],
      "products": [
        {
          "name": "PC Computer",
          "quantity": 1,
          "price": null
        },
        {
          "name": "SSD/NVME Drive",
          "quantity": 1,
          "price": null
        }
      ],
      "emails": [],
      "phones": [],
      "money_amounts": [],
      "dates": ["2024-06-16"]
    },
    "intent": "customer_support",
    "sentiment": "negative",
    "confidence_score": 0.85
  },
  "error": null,
  "latency": 2.3,
  "metadata": {
    "file_name": "in-19524528884-6366754074-20250616-190907-1750118947.21929.wav",
    "duration_seconds": 2465.64,
    "word_count": 627,
    "original_model": "nvidia/canary-1b-flash"
  }
}
```

### 2. Common Patterns in Customer Support Data

#### **Intents You'll See**:
- `customer_support` (most common)
- `general_discussion`
- `project_update` (for internal calls)

#### **Sentiments Expected**:
- `negative` (frustrated customers with issues)
- `neutral` (standard support interactions)
- `positive` (resolved issues, satisfied customers)

#### **Action Items Typically Found**:
- Follow-up calls scheduled
- Technical troubleshooting steps
- Replacement part orders
- Escalation to specialists
- Customer callbacks

#### **Entities Likely Extracted**:
- **Products**: Computer models, software, hardware components
- **Dates**: Callback dates, purchase dates, warranty expiration
- **Phone numbers**: Customer contact numbers
- **Technical identifiers**: Serial numbers, order IDs
- **Money amounts**: Refund amounts, service costs

### 3. Performance Metrics You Should Expect

Based on customer support transcripts, expect:

```json
{
  "total_requests": 100,
  "successful_extractions": 85,
  "failed_extractions": 15,
  "retry_count": 12,
  "avg_latency": 2.1,
  "json_pass_rate": 0.85,
  "field_hit_rates": {
    "summary": 0.95,
    "action_items": 0.75,
    "invoice_ids": 0.05,
    "order_ids": 0.15,
    "products": 0.65,
    "emails": 0.10,
    "phones": 0.20,
    "money_amounts": 0.25,
    "dates": 0.40
  }
}
```

**Field Hit Rate Explanation**:
- **High hit rates** (>70%): summary, action_items, products
- **Medium hit rates** (20-70%): dates, money_amounts, phones
- **Low hit rates** (<20%): invoice_ids, order_ids, emails

### 4. Common Challenges

#### **Why Some Extractions May Fail**:
1. **Transcription Quality**: Speech-to-text errors in original transcription
2. **Background Noise**: Conversations with children, multiple speakers
3. **Technical Jargon**: Complex technical terms may be misunderstood
4. **Informal Speech**: "Yeah", "um", conversational patterns

#### **Retry Scenarios**:
- Invalid JSON format from model
- Missing required fields
- Schema validation errors

### 5. Sample Run Commands

**Start vLLM Server** (in separate terminal):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16000 \
  --trust-remote-code \
  --port 8000
```

**Process Transcriptions**:
```bash
# Test with first 5 records
python process_transcriptions.py

# Or use the action pipeline directly
python action_pipeline.py
```

### 6. Output Files Generated

1. **transcription_results_sample.json**: Processed results
2. **Console output**: Real-time progress and metrics
3. **Error logs**: Failed extractions with reasons

### 7. Optimization Expectations

The pipeline optimizes for speed by:
- **Filtering salient lines**: Only processes relevant content
- **Conservative token limits**: 400-600 tokens max
- **Regex hints**: Helps model find specific patterns
- **Temperature 0.2**: Consistent, focused outputs

This should give you reliable, structured data from your customer support transcriptions suitable for analysis, reporting, and automation.