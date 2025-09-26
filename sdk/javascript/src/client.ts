/**
 * S2A JavaScript SDK Client Implementation
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import FormData from 'form-data';
import * as fs from 'fs';
import * as path from 'path';
import * as mime from 'mime-types';

import {
  S2AClientConfig,
  TranscriptionResult,
  IntelligenceResult,
  QuickIntelligenceResult,
  CompleteResult,
  AsyncJob,
  JobStatus,
  AudioInput,
  TranscribeOptions,
  TranscribeAsyncOptions,
  IntelligenceOptions,
  AudioValidation
} from './types';

import {
  JobStatusType,
  IntelligenceMode,
  Priority
} from './enums';

import {
  S2AError,
  AuthenticationError,
  RateLimitError,
  AudioValidationError,
  TimeoutError,
  IntelligenceUnavailableError
} from './errors';

const DEFAULT_BASE_URL = 'https://api.bytepulseai.com';
const DEFAULT_TIMEOUT = 300000; // 5 minutes
const MAX_SYNC_AUDIO_DURATION = 120; // 2 minutes
const MAX_ASYNC_AUDIO_DURATION = 7200; // 2 hours

export class S2AClient {
  private readonly apiKey: string;
  private readonly httpClient: AxiosInstance;
  private readonly config: Required<S2AClientConfig>;

  constructor(config: S2AClientConfig) {
    if (!config.apiKey) {
      throw new Error('API key is required');
    }

    if (!config.apiKey.match(/^bp-(proj-|svc-)?[a-zA-Z0-9]+/)) {
      throw new Error('Invalid API key format. Must start with bp-proj-, bp-, or bp-svc-');
    }

    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl || DEFAULT_BASE_URL,
      timeout: config.timeout || DEFAULT_TIMEOUT,
      maxRetries: config.maxRetries || 3,
      retryDelay: config.retryDelay || 1000
    };

    this.apiKey = this.config.apiKey;

    // Initialize HTTP client
    this.httpClient = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'User-Agent': 'S2A-JavaScript-SDK/1.0.0'
      }
    });

    // Setup response interceptor for error handling
    this.httpClient.interceptors.response.use(
      (response) => response,
      (error) => this.handleResponseError(error)
    );
  }

  /**
   * Synchronous audio transcription (max 2 minutes)
   */
  async transcribe(
    audioFile: AudioInput,
    options: TranscribeOptions = {}
  ): Promise<TranscriptionResult> {
    const validation = await this.validateAudio(audioFile);

    if (validation.duration && validation.duration > MAX_SYNC_AUDIO_DURATION) {
      throw new AudioValidationError(
        `Audio duration (${validation.duration}s) exceeds sync API limit (${MAX_SYNC_AUDIO_DURATION}s). Use transcribeAsync() for longer audio files.`
      );
    }

    const formData = await this.prepareAudioFormData(audioFile);
    formData.append('enhance_audio', String(options.enhanceAudio ?? true));
    formData.append('remove_silence', String(options.removeSilence ?? false));

    const response = await this.httpClient.post('/v1/transcription/transcribe', formData, {
      headers: formData.getHeaders?.() || { 'Content-Type': 'multipart/form-data' }
    });

    return this.parseTranscriptionResponse(response.data);
  }

  /**
   * Asynchronous audio transcription (max 2 hours)
   */
  async transcribeAsync(
    audioFile: AudioInput,
    options: TranscribeAsyncOptions
  ): Promise<AsyncJob> {
    const validation = await this.validateAudio(audioFile);

    if (validation.duration && validation.duration > MAX_ASYNC_AUDIO_DURATION) {
      throw new AudioValidationError(
        `Audio duration (${validation.duration}s) exceeds async API limit (${MAX_ASYNC_AUDIO_DURATION}s).`
      );
    }

    const formData = await this.prepareAudioFormData(audioFile);
    formData.append('callback_url', options.callbackUrl);
    formData.append('enhance_audio', String(options.enhanceAudio ?? true));
    formData.append('remove_silence', String(options.removeSilence ?? false));
    formData.append('priority', options.priority ?? Priority.NORMAL);

    const response = await this.httpClient.post('/v1/transcription/transcribe_async', formData, {
      headers: formData.getHeaders?.() || { 'Content-Type': 'multipart/form-data' }
    });

    return this.parseAsyncJobResponse(response.data);
  }

  /**
   * Extract comprehensive business intelligence from transcript
   */
  async extractIntelligence(
    transcript: string,
    options: IntelligenceOptions = {}
  ): Promise<IntelligenceResult> {
    const data = {
      transcript_id: `sdk_${Date.now()}`,
      transcript_text: transcript,
      mode: options.mode || IntelligenceMode.AUTO_DETECT
    };

    try {
      const response = await this.httpClient.post('/v1/intelligence/extract/sync', data);
      return this.parseIntelligenceResponse(response.data.intelligence);
    } catch (error) {
      if (error instanceof S2AError && [503, 502].includes(error.statusCode || 0)) {
        throw new IntelligenceUnavailableError('Intelligence service temporarily unavailable');
      }
      throw error;
    }
  }

  /**
   * Extract quick intelligence insights (1-2 seconds)
   */
  async extractQuickIntelligence(transcript: string): Promise<QuickIntelligenceResult> {
    const data = {
      transcript_id: `sdk_quick_${Date.now()}`,
      transcript_text: transcript,
      mode: IntelligenceMode.QUICK
    };

    const response = await this.httpClient.post('/v1/intelligence/extract/sync', data);
    return this.parseQuickIntelligenceResponse(response.data.intelligence);
  }

  /**
   * Transcribe audio and extract intelligence in one call
   */
  async transcribeWithIntelligence(
    audioFile: AudioInput,
    options: TranscribeOptions & IntelligenceOptions & { includeQuick?: boolean } = {}
  ): Promise<CompleteResult> {
    // First transcribe
    const transcription = await this.transcribe(audioFile, options);

    // Then extract intelligence
    let quickIntelligence: QuickIntelligenceResult | undefined;
    let enhancedIntelligence: IntelligenceResult | undefined;

    try {
      if (options.includeQuick !== false) {
        quickIntelligence = await this.extractQuickIntelligence(transcription.text);
      }

      enhancedIntelligence = await this.extractIntelligence(transcription.text, options);
    } catch (error) {
      if (!(error instanceof IntelligenceUnavailableError)) {
        throw error;
      }
      // Continue without intelligence if service is unavailable
    }

    return {
      transcription,
      quickIntelligence,
      enhancedIntelligence
    };
  }

  /**
   * Asynchronous transcription with automatic intelligence extraction
   */
  async transcribeAsyncWithIntelligence(
    audioFile: AudioInput,
    options: TranscribeAsyncOptions & IntelligenceOptions & { includeIntelligence?: boolean } = {
      callbackUrl: ''
    }
  ): Promise<AsyncJob> {
    // Add intelligence parameters to callback URL
    const intelligenceMode = options.mode || IntelligenceMode.AUTO_DETECT;
    const includeIntelligence = options.includeIntelligence ?? true;
    const enhancedCallback = `${options.callbackUrl}?intelligence_mode=${intelligenceMode}&include_intelligence=${includeIntelligence}`;

    return this.transcribeAsync(audioFile, {
      ...options,
      callbackUrl: enhancedCallback
    });
  }

  /**
   * Get status of async job
   */
  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await this.httpClient.get(`/v1/transcription/status/${jobId}`);
    return this.parseJobStatusResponse(response.data);
  }

  /**
   * Wait for async job completion and return results
   */
  async waitForCompletion(
    jobId: string,
    options: { timeout?: number; pollInterval?: number } = {}
  ): Promise<CompleteResult> {
    const timeout = options.timeout || this.config.timeout;
    const pollInterval = options.pollInterval || 5000;
    const startTime = Date.now();

    while (true) {
      const status = await this.getJobStatus(jobId);

      if (status.status === JobStatusType.COMPLETED) {
        const response = await this.httpClient.get(`/v1/transcription/result/${jobId}`);
        return this.parseCompleteResultResponse(response.data);
      }

      if (status.status === JobStatusType.FAILED) {
        throw new S2AError(`Job failed: ${status.errorMessage}`);
      }

      if (Date.now() - startTime > timeout) {
        throw new TimeoutError(`Job ${jobId} did not complete within ${timeout}ms`);
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
  }

  /**
   * Validate audio file without processing
   */
  async validateAudio(audioFile: AudioInput): Promise<AudioValidation> {
    try {
      if (typeof audioFile === 'string') {
        // File path
        const stats = fs.statSync(audioFile);
        const mimeType = mime.lookup(audioFile) || 'application/octet-stream';

        return {
          valid: true,
          fileSize: stats.size,
          mimeType,
          format: path.extname(audioFile).substring(1)
        };
      } else if (audioFile instanceof Buffer) {
        // Buffer
        return {
          valid: true,
          fileSize: audioFile.length,
          mimeType: 'application/octet-stream',
          format: 'unknown'
        };
      } else if (typeof File !== 'undefined' && audioFile instanceof File) {
        // Browser File object
        return {
          valid: true,
          fileSize: audioFile.size,
          mimeType: audioFile.type,
          format: audioFile.name.split('.').pop() || 'unknown'
        };
      } else {
        // Blob or other
        return {
          valid: true,
          mimeType: 'application/octet-stream',
          format: 'unknown'
        };
      }
    } catch (error) {
      throw new AudioValidationError(`Audio validation failed: ${error}`);
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<Record<string, any>> {
    try {
      const response = await this.httpClient.get('/v1/statistics/health');
      return response.data;
    } catch (error) {
      return { status: 'unhealthy', error: error.toString() };
    }
  }

  // Private helper methods

  private async prepareAudioFormData(audioFile: AudioInput): Promise<FormData> {
    const formData = new FormData();

    if (typeof audioFile === 'string') {
      // File path
      const stream = fs.createReadStream(audioFile);
      const filename = path.basename(audioFile);
      formData.append('audio_file', stream, filename);
    } else if (audioFile instanceof Buffer) {
      // Buffer
      formData.append('audio_file', audioFile, 'audio.wav');
    } else if (typeof File !== 'undefined' && audioFile instanceof File) {
      // Browser File object
      formData.append('audio_file', audioFile as any, audioFile.name);
    } else {
      // Blob or other
      formData.append('audio_file', audioFile as any, 'audio.wav');
    }

    return formData;
  }

  private handleResponseError(error: any): never {
    if (error.response) {
      const { status, data } = error.response;

      switch (status) {
        case 401:
          throw new AuthenticationError('Invalid API key or insufficient permissions');
        case 429:
          const retryAfter = parseInt(error.response.headers['retry-after'] || '60');
          throw new RateLimitError('Rate limit exceeded', retryAfter);
        case 413:
          throw new AudioValidationError('Audio file too large');
        case 422:
          throw new AudioValidationError(`Audio validation failed: ${data?.detail || 'Unknown error'}`);
        default:
          throw new S2AError(
            data?.detail || `HTTP ${status} error`,
            status,
            data
          );
      }
    } else if (error.code === 'ECONNABORTED') {
      throw new TimeoutError('Request timeout');
    } else {
      throw new S2AError(error.message || 'Unknown error');
    }
  }

  private parseTranscriptionResponse(data: any): TranscriptionResult {
    return {
      jobId: data.job_id,
      text: data.text || '',
      duration: data.duration || 0,
      confidence: data.confidence || 0,
      processingTime: data.processing_time || 0,
      rtf: data.rtf || 0,
      chunks: data.chunks || 1,
      audioQuality: data.audio_quality
    };
  }

  private parseAsyncJobResponse(data: any): AsyncJob {
    return {
      jobId: data.job_id,
      status: data.status as JobStatusType,
      createdAt: new Date(data.created_at),
      callbackUrl: data.callback_url,
      priority: data.priority as Priority,
      estimatedCompletion: data.estimated_completion ? new Date(data.estimated_completion) : undefined
    };
  }

  private parseJobStatusResponse(data: any): JobStatus {
    return {
      jobId: data.job_id,
      status: data.status as JobStatusType,
      progressPercent: data.progress_percent,
      processingTime: data.processing_time,
      errorMessage: data.error_message,
      resultAvailable: data.result_available || false
    };
  }

  private parseIntelligenceResponse(data: any): IntelligenceResult {
    return {
      callType: data.call_type || 'internal_meeting',
      intent: data.intent || 'general_discussion',
      sentiment: data.sentiment || 'neutral',
      summary: data.summary || '',
      keyTopics: data.key_topics || [],
      people: data.entities?.people || [],
      companies: data.entities?.companies || [],
      products: data.entities?.products || [],
      actionItems: data.action_items || [],
      emails: data.entities?.emails || [],
      phones: data.entities?.phones || [],
      dates: data.entities?.dates || [],
      financialInfo: data.entities?.financial_info || { amounts: [], currency: 'USD', discountRequests: [] },
      opportunityInfo: data.opportunity_info,
      issues: data.issues || [],
      conversationMetrics: data.conversation_metrics || { totalSpeakers: 0, questionCount: 0, interruptions: 0 },
      confidenceScore: data.confidence_score || 0.8,
      completenessScore: data.completeness_score || 0.8,
      recommendations: data.recommendations || [],
      riskFlags: data.risk_flags || []
    };
  }

  private parseQuickIntelligenceResponse(data: any): QuickIntelligenceResult {
    return {
      summary: data.summary || '',
      intent: data.intent || 'general_discussion',
      sentiment: data.sentiment || 'neutral',
      actionItems: data.action_items || [],
      keyEntities: data.key_entities || [],
      confidenceScore: data.confidence_score || 0.8,
      processingTime: data.processing_time || 0
    };
  }

  private parseCompleteResultResponse(data: any): CompleteResult {
    const result: CompleteResult = {
      transcription: this.parseTranscriptionResponse(data.transcription || data)
    };

    if (data.quick_intelligence) {
      result.quickIntelligence = this.parseQuickIntelligenceResponse(data.quick_intelligence);
    }

    if (data.enhanced_intelligence) {
      result.enhancedIntelligence = this.parseIntelligenceResponse(data.enhanced_intelligence);
    }

    return result;
  }
}