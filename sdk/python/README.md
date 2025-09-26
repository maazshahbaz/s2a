# S2A Python SDK

[![PyPI version](https://badge.fury.io/py/s2a-sdk.svg)](https://badge.fury.io/py/s2a-sdk)
[![Python versions](https://img.shields.io/pypi/pyversions/s2a-sdk.svg)](https://pypi.org/project/s2a-sdk/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the S2A (Speech-to-Actions) Platform - Transform audio into actionable business intelligence.

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install s2a-sdk

# With audio processing support
pip install s2a-sdk[audio]

# Development installation
pip install s2a-sdk[dev]
```

### Basic Usage

```python
from s2a_sdk import S2AClient

# Initialize client
client = S2AClient(api_key="bp-proj-your-api-key")

# Simple transcription
result = client.transcribe("meeting.mp3")
print(f"Transcript: {result.text}")

# Transcription with business intelligence
result = client.transcribe_with_intelligence("sales_call.wav")
print(f"Summary: {result.enhanced_intelligence.summary}")
print(f"Intent: {result.enhanced_intelligence.intent}")
print(f"Action Items: {len(result.enhanced_intelligence.action_items)}")
```

## 🎯 Key Features

### **Multi-Stage Intelligence Extraction**
- **Quick Intelligence** (1-2s): Immediate insights for real-time applications
- **Enhanced Intelligence** (5-15s): Comprehensive 50+ field business analysis
- **Auto-Detection**: Automatically identifies sales, support, or general conversations

### **Comprehensive Business Intelligence**
- **Action Items**: Task extraction with assignees, priorities, and due dates
- **Entity Recognition**: People, companies, products, financial data, contacts
- **Conversation Analysis**: Speaker identification, talk-time, interaction metrics
- **Business Context**: Sales opportunities, support issues, meeting insights

### **Professional SDK Features**
- **Type Safety**: Full typing support with IntelliSense
- **Error Handling**: Automatic retries with exponential backoff
- **Audio Validation**: Built-in format and duration validation
- **Async Support**: Both sync and async processing workflows

## 📚 Documentation

### Core Methods

#### `transcribe(audio_file, enhance_audio=True)`
**Synchronous transcription (≤2 minutes)**
```python
result = client.transcribe("short_audio.wav")
print(f"Text: {result.text}")
print(f"Duration: {result.duration}s")
print(f"Confidence: {result.confidence}")
```

#### `transcribe_async(audio_file, callback_url, priority="normal")`
**Asynchronous transcription (≤2 hours)**
```python
job = client.transcribe_async(
    "long_meeting.mp3",
    callback_url="https://yourapp.com/webhook",
    priority="high"
)
print(f"Job ID: {job.job_id}")

# Wait for completion
result = client.wait_for_completion(job.job_id, timeout=600)
```

#### `transcribe_with_intelligence(audio_file, intelligence_mode="auto_detect")`
**Combined transcription + intelligence**
```python
result = client.transcribe_with_intelligence("sales_call.wav")

# Access transcription
print(f"Transcript: {result.transcription.text}")

# Access quick intelligence (immediate)
if result.quick_intelligence:
    print(f"Quick Summary: {result.quick_intelligence.summary}")

# Access enhanced intelligence (comprehensive)
if result.enhanced_intelligence:
    print(f"Call Type: {result.enhanced_intelligence.call_type}")
    print(f"Key People: {[p.name for p in result.enhanced_intelligence.people]}")
    print(f"Action Items: {len(result.enhanced_intelligence.action_items)}")
```

### Intelligence-Only Methods

#### `extract_intelligence(transcript, mode="auto_detect")`
**Extract comprehensive business intelligence**
```python
intelligence = client.extract_intelligence(transcript_text, mode="sales")

print(f"Intent: {intelligence.intent}")
print(f"Sentiment: {intelligence.sentiment}")

# Sales-specific insights
if intelligence.opportunity_info:
    print(f"Deal Stage: {intelligence.opportunity_info.stage}")
    print(f"Value: ${intelligence.opportunity_info.value_estimate}")

# People mentioned
for person in intelligence.people:
    print(f"- {person.name} ({person.role}) at {person.company}")

# Action items
for item in intelligence.action_items:
    print(f"TODO: {item.task} (assigned to: {item.assignee})")
```

#### `extract_quick_intelligence(transcript)`
**Fast 1-2 second extraction**
```python
quick = client.extract_quick_intelligence(transcript_text)
print(f"Summary: {quick.summary}")
print(f"Top Actions: {[item.task for item in quick.action_items]}")
```

## 🎨 Advanced Examples

### Sales Call Analysis
```python
# Process sales call recording
result = client.transcribe_with_intelligence("sales_demo.mp3", intelligence_mode="sales")

# Extract sales insights
intelligence = result.enhanced_intelligence
if intelligence.opportunity_info:
    print(f"Lead Quality Score: {intelligence.opportunity_info.close_probability}")
    print(f"Timeline: {intelligence.opportunity_info.timeline}")
    print(f"Decision Criteria: {intelligence.opportunity_info.decision_criteria}")

# Competitive mentions
for competitor in intelligence.entities.competitors:
    print(f"Competitor mentioned: {competitor.name} ({competitor.mention_type})")

# Financial discussion
financial = intelligence.entities.financial_info
if financial.budget_range:
    print(f"Budget range: ${financial.budget_range['min']}-${financial.budget_range['max']}")
```

### Customer Support Analysis
```python
result = client.transcribe_with_intelligence("support_call.mp3", intelligence_mode="support")

intelligence = result.enhanced_intelligence

# Support-specific metrics
if hasattr(intelligence, 'support_intelligence'):
    support = intelligence.support_intelligence
    print(f"Escalation Risk: {support.escalation_risk}")
    print(f"Customer Satisfaction: {support.customer_satisfaction}")
    print(f"First Call Resolution: {support.first_call_resolution}")

# Issues identified
for issue in intelligence.issues:
    print(f"Issue: {issue.description} (severity: {issue.severity})")
    if issue.workaround:
        print(f"  Workaround: {issue.workaround}")
```

### Async Processing with Webhooks
```python
import asyncio

async def process_multiple_files():
    """Process multiple audio files concurrently"""
    files = ["meeting1.mp3", "meeting2.mp3", "meeting3.mp3"]
    jobs = []

    # Submit all jobs
    for file in files:
        job = client.transcribe_async_with_intelligence(
            file,
            callback_url=f"https://yourapp.com/webhook/{file}",
            intelligence_mode="auto_detect"
        )
        jobs.append(job)
        print(f"Submitted {file}: {job.job_id}")

    # Monitor completion
    for job in jobs:
        result = client.wait_for_completion(job.job_id)
        print(f"Completed {job.job_id}")

# Run async processing
asyncio.run(process_multiple_files())
```

### Error Handling
```python
from s2a_sdk import S2AClient, AudioValidationError, RateLimitError, AuthenticationError

try:
    result = client.transcribe("large_file.mp3")
except AudioValidationError as e:
    print(f"Audio validation failed: {e}")
    # File too large for sync API, use async instead
    job = client.transcribe_async("large_file.mp3", "https://yourapp.com/webhook")

except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
    time.sleep(e.retry_after)

except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Check your API key
```

### Audio Validation
```python
# Validate audio before processing
validation = client.validate_audio("meeting.mp3")
print(f"Duration: {validation['duration']}s")
print(f"File size: {validation['file_size']} bytes")
print(f"Format: {validation['format']}")

if validation['duration'] > 120:  # 2 minutes
    print("File too long for sync API, using async...")
    job = client.transcribe_async("meeting.mp3", callback_url)
```

## 🔧 Configuration

### Environment Variables
```bash
# Set default API key
export S2A_API_KEY="bp-proj-your-api-key"

# Set custom API base URL
export S2A_BASE_URL="https://your-custom-s2a-instance.com"
```

### Client Configuration
```python
client = S2AClient(
    api_key="bp-proj-your-key",
    base_url="https://api.bytepulseai.com",  # Custom base URL
    timeout=300,  # 5 minute timeout
    max_retries=3,  # Retry failed requests
    retry_delay=1.0  # Initial retry delay
)
```

## 📊 Response Models

### TranscriptionResult
```python
@dataclass
class TranscriptionResult:
    job_id: str
    text: str                    # Transcribed text
    duration: float             # Audio duration in seconds
    confidence: float           # Transcription confidence (0-1)
    processing_time: float      # Processing time in seconds
    rtf: float                  # Real-time factor
    chunks: int                 # Number of audio chunks processed
    audio_quality: Dict         # Audio quality metrics
```

### IntelligenceResult
```python
@dataclass
class IntelligenceResult:
    # Core classification
    call_type: str              # "sales_call", "customer_support", etc.
    intent: str                 # Primary conversation intent
    sentiment: str              # Overall sentiment
    summary: str                # Conversation summary

    # Extracted entities
    people: List[Person]        # People mentioned with roles, companies
    companies: List[str]        # Company names
    products: List[Product]     # Products/services discussed
    action_items: List[ActionItem]  # Tasks with assignees, priorities

    # Contact information
    emails: List[str]           # Email addresses
    phones: List[str]           # Phone numbers
    dates: List[str]            # Important dates

    # Business context
    opportunity_info: Dict      # Sales opportunity details
    issues: List[Dict]          # Support issues identified

    # Analysis
    conversation_metrics: ConversationMetrics  # Talk time, interactions
    recommendations: List[str]   # AI recommendations
    confidence_score: float     # Overall extraction confidence
```

## 🚨 Error Types

- **`AudioValidationError`**: Invalid audio file or format
- **`AuthenticationError`**: Invalid API key or permissions
- **`RateLimitError`**: API rate limit exceeded
- **`TimeoutError`**: Request or processing timeout
- **`IntelligenceUnavailableError`**: Intelligence service unavailable
- **`S2AError`**: Base error class for all SDK errors

## 🔒 Authentication

The SDK supports S2A API keys in the following formats:
- **Project keys**: `bp-proj-*` (recommended for applications)
- **User keys**: `bp-*` (for individual users)
- **Service keys**: `bp-svc-*` (for server-to-server)

Get your API key from the [S2A Dashboard](https://dashboard.bytepulseai.com).

## 📝 Changelog

### Version 1.0.0
- Initial release
- Complete transcription and intelligence features
- Multi-stage intelligence extraction
- Comprehensive business intelligence models
- Full async support
- Audio validation and error handling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://docs.bytepulseai.com](https://docs.bytepulseai.com)
- **API Reference**: [https://api.bytepulseai.com/docs](https://api.bytepulseai.com/docs)
- **Issues**: [GitHub Issues](https://github.com/99technologies-ai/s2a/issues)
- **Email**: support@99technologies.ai