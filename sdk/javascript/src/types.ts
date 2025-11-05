/**
 * Type definitions for S2A SDK
 */

import {
  JobStatusType,
  IntelligenceMode,
  AudioFormat,
  Priority,
  CallType,
  Intent,
  Sentiment
} from './enums';

// Core interfaces
export interface S2AClientConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
}

export interface ActionItem {
  task: string;
  assignee?: string;
  dueDate?: string;
  priority: string;
  confidence: number;
}

export interface Person {
  name: string;
  role?: string;
  company?: string;
  email?: string;
  phone?: string;
  isDecisionMaker?: boolean;
}

export interface Product {
  name: string;
  category?: string;
  quantity?: number;
  price?: number;
  featuresDiscussed: string[];
}

export interface FinancialInfo {
  amounts: number[];
  budgetRange?: { min: number; max: number };
  currency: string;
  discountRequests: number[];
}

export interface ConversationMetrics {
  totalSpeakers: number;
  customerTalkTimePercent?: number;
  agentTalkTimePercent?: number;
  questionCount: number;
  interruptions: number;
  paceRating?: string;
}

// Diarization types
export interface SpeakerSegment {
  speaker: string; // e.g., "SPK_1", "SPK_2"
  start: number;  // Start time in seconds
  end: number;    // End time in seconds
  text: string;   // Transcribed text for this segment
}

export interface DiarizationResult {
  speakerTranscript: SpeakerSegment[];
  numSpeakers: number;
  diarModel: string;  // Model used for diarization
  diarizationStatus: string;  // "completed", "failed", etc.
  audioDuration: number;  // Total audio duration in seconds
}

// Result types
export interface TranscriptionResult {
  jobId: string;
  status: string;
  text: string;
  duration: number;
  rtf: number;
  processingTime: number;
  chunks: number;
  confidence: number;
  audioQuality?: Record<string, any>;
  quickIntelligence?: QuickIntelligenceResult | null;
  enhancedIntelligenceStatus?: IntelligenceResult | null;
  diarization?: DiarizationResult | null;
}

export interface QuickIntelligenceResult {
  summary: string;
  intent: Intent;
  sentiment: Sentiment;
  actionItems: ActionItem[];
  keyEntities: string[];
  confidenceScore: number;
  processingTime: number;
}

export interface IntelligenceResult {
  // Core classification
  callType: CallType;
  intent: Intent;
  sentiment: Sentiment;
  summary: string;
  keyTopics: string[];

  // Extracted entities
  people: Person[];
  companies: string[];
  products: Product[];
  actionItems: ActionItem[];

  // Contact information
  emails: string[];
  phones: string[];
  dates: string[];

  // Financial data
  financialInfo: FinancialInfo;

  // Business context
  opportunityInfo?: Record<string, any>;
  issues: Record<string, any>[];

  // Conversation analysis
  conversationMetrics: ConversationMetrics;

  // Quality scores
  confidenceScore: number;
  completenessScore: number;

  // AI recommendations
  recommendations: string[];
  riskFlags: string[];
}

export interface CompleteResult {
  transcription: TranscriptionResult;
  quickIntelligence?: QuickIntelligenceResult;
  enhancedIntelligence?: IntelligenceResult;
}

export interface AsyncJob {
  jobId: string;
  status: JobStatusType;
}

export interface JobStatus {
  jobId: string;
  status: string;
  error?: string | null;
  result?: TranscriptionResult | null;
}

// Request types
export interface TranscribeOptions {
  enhanceAudio?: boolean;
  removeSilence?: boolean;
}

export interface TranscribeAsyncOptions extends TranscribeOptions {
  callbackUrl: string;
  priority?: Priority;
}

export interface IntelligenceOptions {
  mode?: IntelligenceMode;
}

export interface AudioValidation {
  valid: boolean;
  fileSize?: number;
  mimeType?: string;
  duration?: number;
  format?: string;
}

// Utility types
export type AudioInput = string | Buffer | File | Blob;

export interface WebhookPayload {
  jobId: string;
  status: string;
  intelligenceType?: 'quick' | 'enhanced' | 'transcription';
  intelligence?: Record<string, any>;
  result?: Record<string, any>;
  error?: string;
  timestamp: number;
  processingTime?: number;
}