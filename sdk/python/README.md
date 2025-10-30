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

# With audio processing support (for duration detection)
pip install s2a-sdk[audio]

# Development installation
pip install s2a-sdk[dev]
```

### Basic Usage

```python
from s2a_sdk import S2AClient

# Initialize client
client = S2AClient(api_key="bp-proj-your-api-key")

# Async transcription (for audio between 1 second and 5 hours)
job = client.transcribe_async(
    "meeting.mp3",
    callback_url="https://yourapp.com/webhook"
)

print(f"Job ID: {job.job_id}")

# Wait for completion
result = client.wait_for_completion(job.job_id)
print(f"Transcript: {result.transcription.text}")

# Access intelligence if included
if result.enhanced_intelligence:
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
- **Type Safety**: Full typing support with type hints
- **Error Handling**: Automatic retries with exponential backoff
- **Audio Validation**: Built-in format and duration validation
- **Context Manager**: Clean resource management with context managers
- **Async Support**: Full async/await support for concurrent operations

## 📚 API Documentation

### Core Methods

#### `transcribe_async(audio_file, callback_url, **options)`
**Asynchronous transcription (min 1 second, max 5 hours)**

```python
job = client.transcribe_async(
    "long_meeting.mp3",
    callback_url="https://yourapp.com/webhook",
    priority=Priority.HIGH,
    enhance_audio=True,
    remove_silence=False
)

print(f"Job ID: {job.job_id}")

# Wait for completion
result = client.wait_for_completion(job.job_id, timeout=600, poll_interval=5)
print(f"Text: {result.transcription.text}")
print(f"Duration: {result.transcription.duration}s")
print(f"Confidence: {result.transcription.confidence}")
```

**Parameters:**
- `audio_file` (str | Path | BinaryIO): Path to audio file or file-like object
- `callback_url` (str, required): URL to receive webhook notifications
- `enhance_audio` (bool, optional): Enable audio enhancement (default: `True`)
- `remove_silence` (bool, optional): Remove silence from audio (default: `False`)
- `priority` (Priority, optional): Processing priority - `Priority.LOW`, `Priority.NORMAL` (default), `Priority.HIGH`

**Returns:** `AsyncJob` with `job_id` and `status`

**Raises:**
- `AudioValidationError`: Audio duration < 1 second or > 5 hours, or invalid format
- `AuthenticationError`: Invalid API key
- `RateLimitError`: API rate limit exceeded

#### `transcribe_async_with_intelligence(audio_file, callback_url, **options)`
**Asynchronous transcription with automatic intelligence extraction**

```python
job = client.transcribe_async_with_intelligence(
    "sales_call.mp3",
    callback_url="https://yourapp.com/webhook",
    intelligence_mode=IntelligenceMode.SALES,
    include_intelligence=True,
    priority=Priority.HIGH
)

# Intelligence will be included in webhook callback and result
result = client.wait_for_completion(job.job_id)

# Access transcription
print(f"Transcript: {result.transcription.text}")

# Access intelligence
if result.enhanced_intelligence:
    print(f"Call Type: {result.enhanced_intelligence.call_type}")
    print(f"Summary: {result.enhanced_intelligence.summary}")
```

**Parameters:**
- `audio_file` (str | Path | BinaryIO): Path to audio file or file-like object
- `callback_url` (str, required): URL to receive webhook notifications
- `intelligence_mode` (IntelligenceMode, optional): Intelligence extraction mode (default: `IntelligenceMode.AUTO_DETECT`)
- `include_intelligence` (bool, optional): Include intelligence in results (default: `True`)
- `enhance_audio` (bool, optional): Enable audio enhancement (default: `True`)
- `priority` (Priority, optional): Processing priority (default: `Priority.NORMAL`)

**Intelligence Modes:**
- `IntelligenceMode.AUTO_DETECT`: Automatically detect conversation type (default)
- `IntelligenceMode.SALES`: Optimize for sales calls
- `IntelligenceMode.SUPPORT`: Optimize for customer support
- `IntelligenceMode.GENERAL`: General conversation analysis
- `IntelligenceMode.QUICK`: Fast extraction mode

### Intelligence-Only Methods

#### `extract_intelligence(transcript, mode=IntelligenceMode.AUTO_DETECT)`
**Extract comprehensive business intelligence from existing transcript**

```python
intelligence = client.extract_intelligence(
    transcript_text,
    mode=IntelligenceMode.SALES
)

print(f"Intent: {intelligence.intent}")
print(f"Sentiment: {intelligence.sentiment}")
print(f"Summary: {intelligence.summary}")

# Sales-specific insights
if intelligence.opportunity_info:
    print(f"Deal Stage: {intelligence.opportunity_info['stage']}")
    print(f"Value: ${intelligence.opportunity_info['value_estimate']}")
    print(f"Close Probability: {intelligence.opportunity_info['close_probability']}")

# People mentioned
for person in intelligence.people:
    print(f"- {person.name} ({person.role}) at {person.company}")

# Action items
for item in intelligence.action_items:
    print(f"TODO: {item.task}")
    if item.assignee:
        print(f"  Assigned to: {item.assignee}")
    if item.due_date:
        print(f"  Due: {item.due_date}")
    print(f"  Priority: {item.priority}")
```

**Returns:** `IntelligenceResult` with comprehensive business intelligence

**Raises:**
- `IntelligenceUnavailableError`: Intelligence service temporarily unavailable

#### `extract_quick_intelligence(transcript)`
**Fast 1-2 second extraction for immediate insights**

```python
quick = client.extract_quick_intelligence(transcript_text)

print(f"Summary: {quick.summary}")
print(f"Intent: {quick.intent}")
print(f"Sentiment: {quick.sentiment}")
print(f"Processing Time: {quick.processing_time}s")
print(f"Confidence: {quick.confidence_score}")

# Action items
for item in quick.action_items:
    print(f"- {item.task} (Priority: {item.priority})")

# Key entities
print(f"Key Entities: {', '.join(quick.key_entities)}")
```

**Returns:** `QuickIntelligenceResult` with basic insights

### Job Management

#### `get_job_status(job_id)`
**Check the status of an async job**

```python
status = client.get_job_status("job_123456")

print(f"Status: {status.status}")
print(f"Job ID: {status.job_id}")

if status.status == JobStatusType.COMPLETED and status.result:
    print(f"Transcript: {status.result.text}")
    print(f"Duration: {status.result.duration}s")
elif status.status == JobStatusType.FAILED:
    print(f"Error: {status.error}")
```

**Returns:** `JobStatus` with current job state

#### `wait_for_completion(job_id, timeout=None, poll_interval=5.0)`
**Wait for job completion and return results**

```python
# Wait with default timeout (300 seconds)
result = client.wait_for_completion("job_123456")

# Wait with custom timeout and poll interval
result = client.wait_for_completion(
    "job_123456",
    timeout=600,      # Wait up to 10 minutes
    poll_interval=3   # Check every 3 seconds
)

print(f"Transcript: {result.transcription.text}")
print(f"Processing time: {result.transcription.processing_time}s")
print(f"RTF: {result.transcription.rtf}")

# Access intelligence if available
if result.enhanced_intelligence:
    print(f"Summary: {result.enhanced_intelligence.summary}")
```

**Parameters:**
- `job_id` (str): Job ID from async transcription
- `timeout` (float, optional): Maximum wait time in seconds (default: client timeout)
- `poll_interval` (float, optional): How often to check status in seconds (default: 5.0)

**Returns:** `CompleteResult` with transcription and optional intelligence

**Raises:**
- `TimeoutError`: Job didn't complete within timeout
- `S2AError`: Job failed during processing

### Utility Methods

#### `validate_audio(audio_file)`
**Validate audio file before processing**

```python
validation = client.validate_audio("meeting.mp3")

print(f"Valid: {validation['valid']}")
print(f"File size: {validation['file_size']} bytes")
print(f"Format: {validation['format']}")
print(f"MIME type: {validation['mime_type']}")
print(f"Duration: {validation['duration']}s")

# Choose appropriate API based on duration
if validation['duration'] and validation['duration'] > 18000:
    print(f"Audio is {validation['duration']}s - exceeds 5 hour limit!")
elif validation['duration'] and validation['duration'] < 1:
    print(f"Audio is too short - minimum 1 second required")
else:
    # Duration is acceptable for async API
    job = client.transcribe_async("meeting.mp3", callback_url)
```

**Returns:** Dictionary with validation results and metadata

#### `estimate_cost(duration_seconds)`
**Estimate processing cost for audio duration**

```python
cost = client.estimate_cost(3600)  # 1 hour
print(f"Estimated processing time: {cost['estimated_processing_time']}s")
print(f"Tier: {cost['tier']}")
```

#### `health_check()`
**Check API health and connectivity**

```python
health = client.health_check()
print(f"Status: {health['status']}")
```

## 🎨 Advanced Examples

### Sales Call Analysis

```python
from s2a_sdk import S2AClient, IntelligenceMode, Priority

client = S2AClient(api_key="bp-proj-your-key")

# Process sales call recording
job = client.transcribe_async_with_intelligence(
    "sales_demo.mp3",
    callback_url="https://yourapp.com/webhook/sales",
    intelligence_mode=IntelligenceMode.SALES,
    priority=Priority.HIGH
)

# Wait for completion
result = client.wait_for_completion(job.job_id, timeout=600)

# Extract sales insights
intelligence = result.enhanced_intelligence
if intelligence and intelligence.opportunity_info:
    print(f"Lead Quality Score: {intelligence.opportunity_info.get('close_probability')}")
    print(f"Timeline: {intelligence.opportunity_info.get('timeline')}")
    print(f"Decision Criteria: {intelligence.opportunity_info.get('decision_criteria')}")
    print(f"Budget: {intelligence.opportunity_info.get('budget')}")
    print(f"Next Steps: {intelligence.opportunity_info.get('next_steps')}")

# Financial discussion
if intelligence:
    financial = intelligence.financial_info
    if financial.budget_range:
        print(f"Budget range: ${financial.budget_range['min']:,.0f} - ${financial.budget_range['max']:,.0f}")
    
    # Mentioned amounts
    if financial.amounts:
        print(f"Amounts discussed: {', '.join([f'${amt:,.0f}' for amt in financial.amounts])}")

    # Next steps and action items
    for item in intelligence.action_items:
        print(f"Follow-up: {item.task}")
        if item.assignee:
            print(f"  Owner: {item.assignee}")
        if item.due_date:
            print(f"  Due: {item.due_date}")
```

### Customer Support Analysis

```python
# Process support call
job = client.transcribe_async_with_intelligence(
    "support_call.mp3",
    callback_url="https://yourapp.com/webhook/support",
    intelligence_mode=IntelligenceMode.SUPPORT
)

result = client.wait_for_completion(job.job_id)
intelligence = result.enhanced_intelligence

if intelligence:
    # Issues identified
    for issue in intelligence.issues:
        print(f"\nIssue: {issue.get('description')}")
        print(f"  Severity: {issue.get('severity')}")
        print(f"  Category: {issue.get('category')}")
        
        if issue.get('workaround'):
            print(f"  Workaround: {issue['workaround']}")
        
        if issue.get('resolution'):
            print(f"  Resolution: {issue['resolution']}")
        
        if issue.get('root_cause'):
            print(f"  Root Cause: {issue['root_cause']}")

    # Customer satisfaction metrics
    metrics = intelligence.conversation_metrics
    if metrics:
        print(f"\nConversation Metrics:")
        if metrics.customer_talk_time_percent:
            print(f"  Customer talk time: {metrics.customer_talk_time_percent:.1f}%")
        if metrics.agent_talk_time_percent:
            print(f"  Agent talk time: {metrics.agent_talk_time_percent:.1f}%")
        print(f"  Questions asked: {metrics.question_count}")
        print(f"  Interruptions: {metrics.interruptions}")
        if metrics.pace_rating:
            print(f"  Pace: {metrics.pace_rating}")

    # Risk flags
    if intelligence.risk_flags:
        print(f"\n⚠️ Risk Flags:")
        for flag in intelligence.risk_flags:
            print(f"  - {flag}")
```

### Batch Processing with Context Manager

```python
import time
from pathlib import Path

def process_multiple_files():
    """Process multiple audio files with proper resource management"""
    
    files = ["meeting1.mp3", "meeting2.mp3", "meeting3.mp3"]
    
    # Use context manager for automatic cleanup
    with S2AClient(api_key="bp-proj-your-key") as client:
        jobs = []
        
        # Submit all jobs
        for file in files:
            try:
                job = client.transcribe_async_with_intelligence(
                    file,
                    callback_url=f"https://yourapp.com/webhook/{file}",
                    intelligence_mode=IntelligenceMode.AUTO_DETECT,
                    priority=Priority.NORMAL
                )
                jobs.append((file, job))
                print(f"✓ Submitted {file}: {job.job_id}")
            except Exception as e:
                print(f"✗ Failed to submit {file}: {e}")
        
        # Monitor completion
        for filename, job in jobs:
            try:
                print(f"\nWaiting for {filename}...")
                result = client.wait_for_completion(
                    job.job_id,
                    timeout=600,
                    poll_interval=5
                )
                
                print(f"✓ Completed {filename}")
                print(f"  Transcript length: {len(result.transcription.text)} chars")
                print(f"  Duration: {result.transcription.duration:.1f}s")
                print(f"  Processing time: {result.transcription.processing_time:.1f}s")
                print(f"  RTF: {result.transcription.rtf:.3f}")
                
                if result.enhanced_intelligence:
                    print(f"  Intent: {result.enhanced_intelligence.intent}")
                    print(f"  Sentiment: {result.enhanced_intelligence.sentiment}")
                    print(f"  Action items: {len(result.enhanced_intelligence.action_items)}")
                
            except Exception as e:
                print(f"✗ Failed {filename}: {e}")

# Run batch processing
process_multiple_files()
```

### Async/Await Support

```python
import asyncio
from s2a_sdk import S2AClient

async def process_files_concurrently():
    """Process multiple files using async/await"""
    
    async with S2AClient(api_key="bp-proj-your-key") as client:
        files = ["call1.mp3", "call2.mp3", "call3.mp3"]
        
        # Submit all jobs
        jobs = []
        for file in files:
            job = client.transcribe_async(
                file,
                callback_url=f"https://yourapp.com/webhook/{file}"
            )
            jobs.append((file, job))
        
        # Wait for all to complete concurrently
        results = []
        for filename, job in jobs:
            result = client.wait_for_completion(job.job_id)
            results.append((filename, result))
        
        return results

# Run async processing
results = asyncio.run(process_files_concurrently())
for filename, result in results:
    print(f"{filename}: {len(result.transcription.text)} characters")
```

### Extract Intelligence from Existing Transcript

```python
# If you already have a transcript from another source
existing_transcript = """
This is a sales call transcript where we discussed...
"""

# Get quick insights (1-2 seconds)
quick = client.extract_quick_intelligence(existing_transcript)
print(f"Quick Summary: {quick.summary}")
print(f"Sentiment: {quick.sentiment}")
print(f"Processing time: {quick.processing_time:.2f}s")

# Get comprehensive analysis (5-15 seconds)
intelligence = client.extract_intelligence(
    existing_transcript,
    mode=IntelligenceMode.SALES
)

print(f"\nCall Type: {intelligence.call_type}")
print(f"Intent: {intelligence.intent}")
print(f"Sentiment: {intelligence.sentiment}")
print(f"Confidence: {intelligence.confidence_score:.2%}")

# Export action items as checklist
print("\n## Action Items")
for item in intelligence.action_items:
    assignee = f" (@{item.assignee})" if item.assignee else ""
    due = f" - Due: {item.due_date}" if item.due_date else ""
    print(f"- [ ] {item.task}{assignee}{due} [{item.priority}]")

# Export key contacts
print("\n## Key Contacts")
for person in intelligence.people:
    contact_info = []
    if person.email:
        contact_info.append(f"📧 {person.email}")
    if person.phone:
        contact_info.append(f"📱 {person.phone}")
    
    role_info = f" - {person.role}" if person.role else ""
    company_info = f" at {person.company}" if person.company else ""
    contacts = f" ({', '.join(contact_info)})" if contact_info else ""
    
    print(f"- **{person.name}**{role_info}{company_info}{contacts}")
```

### Comprehensive Error Handling

```python
from s2a_sdk import (
    S2AClient,
    AudioValidationError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    IntelligenceUnavailableError,
    S2AError
)
import time

def robust_transcription(audio_file: str, callback_url: str):
    """Transcribe with comprehensive error handling"""
    
    client = S2AClient(api_key="bp-proj-your-key")
    
    try:
        # Validate first
        validation = client.validate_audio(audio_file)
        print(f"Audio validated: {validation['duration']:.1f}s")
        
        # Check duration limits
        duration = validation.get('duration')
        if duration and duration < 1:
            print(f"❌ Audio too short: {duration:.1f}s (minimum: 1s)")
            return None
        elif duration and duration > 18000:  # 5 hours
            print(f"❌ Audio too long: {duration/3600:.1f}h (maximum: 5h)")
            return None
        
        # Submit job
        job = client.transcribe_async(
            audio_file,
            callback_url=callback_url,
            enhance_audio=True
        )
        print(f"✓ Job submitted: {job.job_id}")
        
        # Wait for completion
        result = client.wait_for_completion(job.job_id, timeout=600)
        print(f"✓ Transcription complete")
        return result
        
    except AudioValidationError as e:
        print(f"❌ Audio validation failed: {e}")
        print("   Check file format, size, or duration")
        return None
        
    except RateLimitError as e:
        print(f"⏸️  Rate limit exceeded. Retry after {e.retry_after} seconds")
        time.sleep(e.retry_after)
        # Retry logic here
        return None
        
    except AuthenticationError as e:
        print(f"🔒 Authentication failed: {e}")
        print("   Check your API key")
        return None
        
    except TimeoutError as e:
        print(f"⏱️  Timeout: {e}")
        print("   Try with longer timeout or check job status manually")
        # Could still retrieve result later with get_job_status
        return None
        
    except IntelligenceUnavailableError as e:
        print(f"⚠️  Intelligence unavailable: {e}")
        print("   Retry later or use transcription only")
        return None
        
    except S2AError as e:
        print(f"❌ SDK Error: {e}")
        if e.status_code:
            print(f"   Status code: {e.status_code}")
        if e.response_data:
            print(f"   Details: {e.response_data}")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
    
    finally:
        client.close()

# Use the robust function
result = robust_transcription("meeting.mp3", "https://yourapp.com/webhook")
if result:
    print(f"Success! Transcript: {result.transcription.text[:100]}...")
```

### Audio Validation Before Processing

```python
def smart_audio_processing(audio_file: str):
    """Validate and choose optimal processing strategy"""
    
    with S2AClient(api_key="bp-proj-your-key") as client:
        # Validate audio
        validation = client.validate_audio(audio_file)
        
        if not validation['valid']:
            print("Invalid audio file")
            return
        
        duration = validation.get('duration')
        file_size = validation.get('file_size', 0)
        
        print(f"Audio file: {audio_file}")
        print(f"  Format: {validation['format']}")
        print(f"  MIME type: {validation['mime_type']}")
        print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
        
        if duration:
            print(f"  Duration: {duration / 60:.1f} minutes ({duration:.1f}s)")
            
            # Check duration limits
            if duration < 1:
                print("\n❌ Audio too short (< 1 second)")
                print("   Minimum duration: 1 second")
                return
            
            elif duration > 18000:  # 5 hours
                print(f"\n❌ Audio too long ({duration / 3600:.1f} hours)")
                print("   Maximum duration: 5 hours")
                return
            
            else:
                # Duration is acceptable - use async API
                print(f"\n✓ Using async API (duration: {duration:.1f}s)")
                
                job = client.transcribe_async_with_intelligence(
                    audio_file,
                    callback_url="https://yourapp.com/webhook",
                    intelligence_mode=IntelligenceMode.AUTO_DETECT
                )
                
                print(f"Job submitted: {job.job_id}")
                return job
        else:
            print("\n⚠️  Duration unknown - proceeding with async API")
            job = client.transcribe_async(
                audio_file,
                callback_url="https://yourapp.com/webhook"
            )
            return job

# Use smart processing
job = smart_audio_processing("my_audio.mp3")
```

## 🔧 Configuration

### Environment Variables

```bash
# Set default API key
export S2A_API_KEY="bp-proj-your-api-key"

# Set custom API base URL (optional)
export S2A_BASE_URL="https://your-custom-s2a-instance.com"
```

```python
import os
from s2a_sdk import S2AClient

# Client will use environment variables if not provided
client = S2AClient(api_key=os.getenv("S2A_API_KEY"))
```

### Client Configuration

```python
from s2a_sdk import S2AClient

client = S2AClient(
    api_key="bp-proj-your-key",              # Required
    base_url="https://api.bytepulseai.com",  # Optional, default API URL
    timeout=300,                              # Optional, 5 minute timeout (default)
    max_retries=3,                            # Optional, retry failed requests (default: 3)
    retry_delay=1.0                           # Optional, initial retry delay in seconds (default: 1.0)
)
```

**Configuration Options:**
- `api_key` (str, required): Your S2A API key (bp-proj-*, bp-*, or bp-svc-*)
- `base_url` (str, optional): Custom API base URL (default: https://api.bytepulseai.com)
- `timeout` (float, optional): Request timeout in seconds (default: 300)
- `max_retries` (int, optional): Number of retry attempts for failed requests (default: 3)
- `retry_delay` (float, optional): Initial delay between retries in seconds (default: 1.0)

### Using Context Managers

```python
# Automatic resource cleanup
with S2AClient(api_key="bp-proj-your-key") as client:
    result = client.transcribe_async("audio.mp3", "https://callback.url")
    # Client automatically closed when exiting context

# Async context manager
async with S2AClient(api_key="bp-proj-your-key") as client:
    result = await client.transcribe_async("audio.mp3", "https://callback.url")
```

## 📊 Response Models

### TranscriptionResult

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TranscriptionResult:
    job_id: str                              # Unique job identifier
    text: str                                # Transcribed text
    duration: float                          # Audio duration in seconds
    confidence: float                        # Transcription confidence (0-1)
    processing_time: float                   # Processing time in seconds
    rtf: float                               # Real-time factor
    status: str                              # Job status
    chunks: int = 1                          # Number of audio chunks processed
    audio_quality: Optional[Dict[str, Any]] = None  # Audio quality metrics
```

### QuickIntelligenceResult

```python
@dataclass
class QuickIntelligenceResult:
    summary: str                             # Brief conversation summary
    intent: str                              # Primary intent
    sentiment: str                           # Overall sentiment
    action_items: List[ActionItem]           # Extracted action items
    key_entities: List[str]                  # Key entities mentioned
    confidence_score: float                  # Extraction confidence (0-1)
    processing_time: float                   # Processing time in seconds
```

### IntelligenceResult

```python
@dataclass
class IntelligenceResult:
    # Core classification
    call_type: str                           # "sales_call", "customer_support", etc.
    intent: str                              # Primary conversation intent
    sentiment: str                           # Overall sentiment
    summary: str                             # Conversation summary
    key_topics: List[str]                    # Main topics discussed

    # Extracted entities
    people: List[Person]                     # People mentioned with roles, companies
    companies: List[str]                     # Company names
    products: List[Product]                  # Products/services discussed
    action_items: List[ActionItem]           # Tasks with assignees, priorities

    # Contact information
    emails: List[str]                        # Email addresses
    phones: List[str]                        # Phone numbers
    dates: List[str]                         # Important dates

    # Financial data
    financial_info: FinancialInfo            # Budget, amounts, discounts

    # Business context
    opportunity_info: Optional[Dict[str, Any]] = None  # Sales opportunity details
    issues: List[Dict[str, Any]] = None      # Support issues identified

    # Conversation analysis
    conversation_metrics: ConversationMetrics = None  # Talk time, interactions

    # Quality scores
    confidence_score: float = 0.8            # Overall extraction confidence
    completeness_score: float = 0.8          # Data completeness score

    # AI recommendations
    recommendations: List[str] = None        # AI recommendations
    risk_flags: List[str] = None             # Potential risks identified
```

### CompleteResult

```python
@dataclass
class CompleteResult:
    transcription: TranscriptionResult
    quick_intelligence: Optional[QuickIntelligenceResult] = None
    enhanced_intelligence: Optional[IntelligenceResult] = None

    @property
    def has_intelligence(self) -> bool:
        """Check if any intelligence data is available"""
        return self.quick_intelligence is not None or self.enhanced_intelligence is not None

    @property
    def best_intelligence(self) -> Union[IntelligenceResult, QuickIntelligenceResult, None]:
        """Get the most comprehensive intelligence available"""
        if self.enhanced_intelligence:
            return self.enhanced_intelligence
        return self.quick_intelligence
```

### AsyncJob

```python
@dataclass
class AsyncJob:
    job_id: str                              # Unique job identifier
    status: JobStatusType                    # Job status enum
```

### JobStatus

```python
@dataclass
class JobStatus:
    job_id: str                              # Unique job identifier
    status:

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

### Version 1.0.4
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