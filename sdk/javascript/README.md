# S2A JavaScript/TypeScript SDK

[![npm version](https://badge.fury.io/js/%4099technologies%2Fs2a-sdk.svg)](https://badge.fury.io/js/%4099technologies%2Fs2a-sdk)
[![Node.js versions](https://img.shields.io/node/v/@99technologies/s2a-sdk.svg)](https://www.npmjs.com/package/@99technologies/s2a-sdk)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official JavaScript/TypeScript SDK for the S2A (Speech-to-Actions) Platform - Transform audio into actionable business intelligence.

## 🚀 Quick Start

### Installation

```bash
# npm
npm install @99technologies/s2a-sdk

# yarn
yarn add @99technologies/s2a-sdk

# pnpm
pnpm add @99technologies/s2a-sdk
```

### Basic Usage

```typescript
import { S2AClient } from '@99technologies/s2a-sdk';

// Initialize client
const client = new S2AClient({
  apiKey: 'bp-proj-your-api-key'
});

// Async transcription (for audio > 2 minutes or up to 2 hours)
const job = await client.transcribeAsync('meeting.mp3', {
  callbackUrl: 'https://yourapp.com/webhook',
  enhanceAudio: true
});

console.log(`Job ID: ${job.jobId}`);

// Wait for completion
const result = await client.waitForCompletion(job.jobId);
console.log(`Transcript: ${result.transcription.text}`);

// Access diarization results (speaker attribution)
if (result.transcription.diarization) {
  const diar = result.transcription.diarization;
  console.log(`Speakers detected: ${diar.numSpeakers}`);
  console.log(`Speaker turns: ${diar.speakerTranscript.length}`);
  
  // Display speaker-attributed transcript
  diar.speakerTranscript.forEach((segment, index) => {
    console.log(`${segment.speaker}: ${segment.text}`);
  });
  
  // Get speaking time per speaker
  diar.speakerTranscript.forEach(segment => {
    const duration = segment.end - segment.start;
    console.log(`${segment.speaker}: ${duration.toFixed(1)}s`);
  });
}

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
- **Full TypeScript Support**: Complete type definitions with IntelliSense
- **Error Handling**: Automatic retries with exponential backoff
- **Audio Validation**: Built-in format and duration validation
- **Async Support**: Both sync and async processing workflows
- **Universal Compatibility**: Works in Node.js and modern browsers

## 📚 API Documentation

### Core Methods

#### `transcribeAsync(audioFile, options)`
**Asynchronous transcription (min 1 sec and max 5 hours)**
```typescript```

const job = await client.transcribeAsync('long_meeting.mp3', {
  callbackUrl: 'https://yourapp.com/webhook',
  priority: Priority.HIGH,
  enhanceAudio: true,
  removeSilence: false
});

console.log(`Job ID: ${job.jobId}`);

// Wait for completion
const result = await client.waitForCompletion(job.jobId, { 
  timeout: 600000,  // 10 minutes
  pollInterval: 5000  // Check every 5 seconds
});

console.log(`Transcript: ${result.transcription.text}`);

**Options**
- **callbackUrl** *(required)*: URL to receive webhook notifications  
- **priority**: `Priority.LOW`, `Priority.NORMAL` *(default)*, `Priority.HIGH`, `Priority.URGENT`  
- **enhanceAudio**: Enable audio enhancement *(default: true)*  
- **removeSilence**: Remove silence from audio *(default: false)*

#### `transcribeAsyncWithIntelligence(audioFile, options)`
**Asynchronous transcription with automatic intelligence extraction**

const job = await client.transcribeAsyncWithIntelligence('sales_call.mp3', {
  callbackUrl: 'https://yourapp.com/webhook',
  mode: IntelligenceMode.SALES,
  includeIntelligence: true,
  priority: Priority.HIGH
});

// Intelligence will be included in webhook callback


### Intelligence-Only Methods

#### `extractIntelligence(transcript, options?)`
**Extract comprehensive business intelligence**
```typescript
const intelligence = await client.extractIntelligence(transcriptText, {
  mode: IntelligenceMode.SALES
});

console.log(`Intent: ${intelligence.intent}`);
console.log(`Sentiment: ${intelligence.sentiment}`);

// Sales-specific insights
if (intelligence.opportunityInfo) {
  console.log(`Deal Stage: ${intelligence.opportunityInfo.stage}`);
  console.log(`Value: $${intelligence.opportunityInfo.value_estimate}`);
}

// People mentioned
intelligence.people.forEach(person => {
  console.log(`- ${person.name} (${person.role}) at ${person.company}`);
});

// Action items
intelligence.actionItems.forEach(item => {
  console.log(`TODO: ${item.task} (assigned to: ${item.assignee})`);
});
```

#### `extractQuickIntelligence(transcript)`
**Fast 1-2 second extraction**
```typescript
const quick = await client.extractQuickIntelligence(transcriptText);
console.log(`Summary: ${quick.summary}`);
console.log(`Top Actions: ${quick.actionItems.map(item => item.task)}`);
```

## 🎨 Advanced Examples

### Sales Call Analysis
```typescript
import { IntelligenceMode, Priority } from '@99technologies/s2a-sdk';

// Process sales call recording
const job = await client.transcribeAsyncWithIntelligence('sales_demo.mp3', {
  callbackUrl: 'https://yourapp.com/webhook/sales',
  mode: IntelligenceMode.SALES,
  priority: Priority.HIGH,
  includeIntelligence: true
});

// Wait for completion
const result = await client.waitForCompletion(job.jobId);

// Extract sales insights
const intelligence = result.enhancedIntelligence;
if (intelligence?.opportunityInfo) {
  console.log(`Lead Quality Score: ${intelligence.opportunityInfo.close_probability}`);
  console.log(`Timeline: ${intelligence.opportunityInfo.timeline}`);
  console.log(`Decision Criteria: ${intelligence.opportunityInfo.decision_criteria}`);
  console.log(`Budget: ${intelligence.opportunityInfo.budget}`);
}

// Financial discussion
const financial = intelligence?.financialInfo;
if (financial?.budgetRange) {
  console.log(`Budget range: $${financial.budgetRange.min}-$${financial.budgetRange.max}`);
}

// Next steps
intelligence?.actionItems.forEach(item => {
  console.log(`Follow-up: ${item.task} (Due: ${item.dueDate})`);
});
```

### Customer Support Analysis
```typescript
const job = await client.transcribeAsyncWithIntelligence('support_call.mp3', {
  callbackUrl: 'https://yourapp.com/webhook/support',
  mode: IntelligenceMode.SUPPORT
});

const result = await client.waitForCompletion(job.jobId);
const intelligence = result.enhancedIntelligence;

// Issues identified
intelligence?.issues.forEach(issue => {
  console.log(`Issue: ${issue.description}`);
  console.log(`  Severity: ${issue.severity}`);
  console.log(`  Category: ${issue.category}`);
  if (issue.workaround) {
    console.log(`  Workaround: ${issue.workaround}`);
  }
  if (issue.resolution) {
    console.log(`  Resolution: ${issue.resolution}`);
  }
});

// Customer satisfaction metrics
const metrics = intelligence?.conversationMetrics;
console.log(`Customer talk time: ${metrics?.customerTalkTimePercent}%`);
console.log(`Questions asked: ${metrics?.questionCount}`);
console.log(`Interruptions: ${metrics?.interruptions}`);
```

### Async Processing with Webhooks
```typescript
async function processMultipleFiles() {
  const files = ['meeting1.mp3', 'meeting2.mp3', 'meeting3.mp3'];
  const jobs: AsyncJob[] = [];

  // Submit all jobs
  for (const file of files) {
    const job = await client.transcribeAsyncWithIntelligence(file, {
      callbackUrl: `https://yourapp.com/webhook/${file}`,
      includeIntelligence: true,
      mode: IntelligenceMode.AUTO_DETECT
    });
    jobs.push(job);
    console.log(`Submitted ${file}: ${job.jobId}`);
  }

  // Monitor completion
  for (const job of jobs) {
    try {
      const result = await client.waitForCompletion(job.jobId, {
        timeout: 600000  // 10 minutes
      });
      console.log(`Completed ${job.jobId}`);
      console.log(`Transcript length: ${result.transcription.text.length} chars`);
    } catch (error) {
      console.error(`Failed ${job.jobId}:`, error);
    }
  }
}

processMultipleFiles();
```
### Async Processing with Webhooks
```typescript
// If you already have a transcript from another source
const existingTranscript = "Your existing transcript text here...";

// Get quick insights
const quick = await client.extractQuickIntelligence(existingTranscript);
console.log(`Quick Summary: ${quick.summary}`);

// Get comprehensive analysis
const intelligence = await client.extractIntelligence(existingTranscript, {
  mode: IntelligenceMode.SALES
});

console.log(`Call Type: ${intelligence.callType}`);
console.log(`Intent: ${intelligence.intent}`);
console.log(`Sentiment: ${intelligence.sentiment}`);

// Export action items
intelligence.actionItems.forEach(item => {
  console.log(`- [ ] ${item.task} (@${item.assignee}) - ${item.priority}`);
});
```


### Error Handling
```typescript
import {
  AudioValidationError,
  RateLimitError,
  AuthenticationError,
  TimeoutError,
  IntelligenceUnavailableError
} from '@99technologies/s2a-sdk';

try {
  const job = await client.transcribeAsync('large_file.mp3', {
    callbackUrl: 'https://yourapp.com/webhook'
  });
  
  const result = await client.waitForCompletion(job.jobId);
  console.log('Success:', result.transcription.text);
  
} catch (error) {
  if (error instanceof AudioValidationError) {
    console.error(`Audio validation failed: ${error.message}`);
    // Check file format, size, or duration
    
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limit exceeded. Retry after ${error.retryAfter} seconds`);
    setTimeout(() => {
      // Retry logic here
    }, error.retryAfter * 1000);
    
  } else if (error instanceof AuthenticationError) {
    console.error(`Authentication failed: ${error.message}`);
    // Check your API key
    
  } else if (error instanceof TimeoutError) {
    console.error(`Request timeout: ${error.message}`);
    // Try with longer timeout or check job status manually
    
  } else if (error instanceof IntelligenceUnavailableError) {
    console.error(`Intelligence service unavailable: ${error.message}`);
    // Retry later or use transcription only
    
  } else {
    console.error(`Unexpected error:`, error);
  }
}
```

### Audio Validation
```typescript
// Validate audio before processing
const validation = await client.validateAudio('meeting.mp3');

console.log(`File size: ${validation.fileSize} bytes`);
console.log(`Format: ${validation.format}`);
console.log(`MIME type: ${validation.mimeType}`);

if (!validation.valid) {
  console.error('Invalid audio file');
  process.exit(1);
}

// Duration check (if available)
if (validation.duration) {
  if (validation.duration > 18000) {  // 5 hours
    console.error('Audio exceeds 5 hour limit for async API');
    process.exit(1);
  }
  
  console.log(`Duration: ${Math.floor(validation.duration / 60)}m ${Math.floor(validation.duration % 60)}s`);
}

// Proceed with transcription
const job = await client.transcribeAsync('meeting.mp3', {
  callbackUrl: 'https://yourapp.com/webhook'
});
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
```typescript
const client = new S2AClient({
  apiKey: 'bp-proj-your-key',
  baseUrl: 'https://api.bytepulseai.com',  // Custom base URL
  timeout: 300000,  // 5 minute timeout
  maxRetries: 3,    // Retry failed requests
  retryDelay: 1000  // Initial retry delay in ms
});
```

## 📊 Response Types

### TranscriptionResult
```typescript
interface TranscriptionResult {
  jobId: string;                   // Unique job identifier
  status: string;                  // Job status
  text: string;                    // Transcribed text
  duration: number;                // Audio duration in seconds
  confidence: number;              // Transcription confidence (0-1)
  processingTime: number;          // Processing time in seconds
  rtf: number;                     // Real-time factor
  chunks: number;                  // Number of audio chunks processed
  audioQuality?: Record<string, any>; // Audio quality metrics
  quickIntelligence?: QuickIntelligenceResult | null;
  enhancedIntelligenceStatus?: IntelligenceResult | null;
}
```
### QuickIntelligenceResult
```typescript
interface QuickIntelligenceResult {
  summary: string;                 // Brief conversation summary
  intent: Intent;                  // Primary intent
  sentiment: Sentiment;            // Overall sentiment
  actionItems: ActionItem[];       // Extracted action items
  keyEntities: string[];           // Key entities mentioned
  confidenceScore: number;         // Extraction confidence (0-1)
  processingTime: number;          // Processing time in seconds
}
```

### IntelligenceResult
```typescript
interface IntelligenceResult {
  // Core classification
  callType: CallType;              // "sales_call", "customer_support", etc.
  intent: Intent;                  // Primary conversation intent
  sentiment: Sentiment;            // Overall sentiment
  summary: string;                 // Conversation summary
  keyTopics: string[];             // Main topics discussed

  // Extracted entities
  people: Person[];                // People mentioned with roles, companies
  companies: string[];             // Company names
  products: Product[];             // Products/services discussed
  actionItems: ActionItem[];       // Tasks with assignees, priorities

  // Contact information
  emails: string[];                // Email addresses
  phones: string[];                // Phone numbers
  dates: string[];                 // Important dates

  // Financial data
  financialInfo: FinancialInfo;    // Budget, amounts, discounts

  // Business context
  opportunityInfo?: Record<string, any>; // Sales opportunity details
  issues: Record<string, any>[];   // Support issues identified

  // Conversation analysis
  conversationMetrics: ConversationMetrics; // Talk time, interactions

  // Quality scores
  confidenceScore: number;         // Overall extraction confidence
  completenessScore: number;       // Data completeness score

  // AI recommendations
  recommendations: string[];       // AI recommendations
  riskFlags: string[];             // Potential risks identified
}
```

### CompleteResult
```typescript
interface CompleteResult {
  transcription: TranscriptionResult;
  quickIntelligence?: QuickIntelligenceResult;
  enhancedIntelligence?: IntelligenceResult;
}
```

### AsyncJob
```typescript
interface AsyncJob {
  jobId: string;                   // Unique job identifier
  status: JobStatusType;           // Job status
}
```

### JobStatus
```typescript
interface JobStatus {
  jobId: string;
  status: string;                  // 'pending', 'processing', 'completed', 'failed'
  error?: string | null;           // Error message if failed
  result?: TranscriptionResult | null; // Result if completed
}
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

## 🌐 Browser Support

The SDK works in modern browsers with support for:
- **File API**: For handling audio file uploads
- **FormData**: For multipart/form-data requests
- **Fetch API**: For HTTP requests (or axios polyfill)

```html
<!-- Include via CDN -->
<script src="https://unpkg.com/@99technologies/s2a-sdk@latest/dist/index.js"></script>

<script>
const client = new S2A.S2AClient({
  apiKey: 'bp-proj-your-key'
});

// Handle file input
document.getElementById('audioFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  try {
    const result = await client.transcribe(file);
    console.log('Transcription:', result.text);
  } catch (error) {
    console.error('Error:', error.message);
  }
});
</script>
```

## 🏗️ Build and Development

```bash
# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Lint code
npm run lint

# Format code
npm run format
```

## 📝 Changelog

### Version 1.0.4
- Initial release
- Complete transcription and intelligence features
- Multi-stage intelligence extraction
- Comprehensive business intelligence models
- Full async support
- Audio validation and error handling
- TypeScript support with full type definitions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://docs.bytepulseai.com](https://docs.bytepulseai.com)
- **API Reference**: [https://api.bytepulseai.com/docs](https://api.bytepulseai.com/docs)
- **Issues**: [GitHub Issues](https://github.com/99technologies-ai/s2a/issues)
- **Email**: support@99technologies.ai